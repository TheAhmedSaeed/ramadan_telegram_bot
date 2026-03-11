import os
import re
import tempfile
import asyncio
import logging

from dotenv import load_dotenv
from google import genai
from google.genai import types
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.error import BadRequest
from telegram.ext import (
    ApplicationBuilder,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)
from pyrogram import Client as PyroClient

load_dotenv()

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
TELEGRAM_API_ID = int(os.environ["TELEGRAM_API_ID"])
TELEGRAM_API_HASH = os.environ["TELEGRAM_API_HASH"]

gemini_client = genai.Client(api_key=GEMINI_API_KEY)
GEMINI_MODEL = "gemini-2.5-flash"

GEMINI_MAX_RETRIES = 3
GEMINI_RETRY_DELAY = 5  # seconds

pyro_client = PyroClient(
    "bot_session",
    api_id=TELEGRAM_API_ID,
    api_hash=TELEGRAM_API_HASH,
    bot_token=TELEGRAM_BOT_TOKEN,
    workdir=os.path.dirname(os.path.abspath(__file__)),
)

# Per-user session storage
user_sessions = {}

# Store last answer for copy button
user_last_answer = {}

SYSTEM_PROMPT = (
    "You have been given a video to analyze. Study it thoroughly — "
    "every visual detail, text on screen, audio, dialogue, actions, transitions, "
    "and context. The user will ask you multiple questions about this video. "
    "Answer each question directly and concisely based on what you observed. "
    "Always respond in the same language as the user's question.\n\n"
    "IMPORTANT style rules:\n"
    "- Write like a knowledgeable human, not an AI assistant\n"
    "- Never use filler phrases like 'بالتأكيد', 'بالطبع', 'نعم،', 'Great question'\n"
    "- Never start with 'بناءً على الفيديو' or 'في الفيديو' — just give the answer directly\n"
    "- No bullet points or numbered lists unless the question asks for a list\n"
    "- Keep it natural and conversational, like a friend explaining something\n"
    "- No bold text, no headers, no formatting — just plain text\n"
    "- Be brief. One short paragraph max unless more detail is truly needed"
)

INSIGHTS_PROMPT = (
    "You just finished analyzing this video. Now provide exactly 10 key insights "
    "from this video. Be very clever in your selection. Write from the perspective "
    "of a Muslim human — what are the most valuable, practical takeaways that "
    "someone can apply directly to their daily life and faith? "
    "Number them 1-10. Be concise but impactful. "
    "Respond in the same language as the video's primary spoken language."
)


def extract_question(text: str) -> str:
    """Extract the actual question from a forwarded message."""
    lines = text.strip().split("\n")
    cleaned = []
    skip_patterns = [
        r"السؤال التفاعلي",
        r"ترسل الإجابة",
        r"@\w+",
        r"^\*\*.*\*\*$",
        r"^\s*$",
    ]
    for line in lines:
        stripped = line.strip()
        if any(re.search(p, stripped) for p in skip_patterns):
            continue
        cleaned.append(stripped)

    result = "\n".join(cleaned).strip()
    return result if result else text.strip()


async def safe_edit_text(msg, text, **kwargs):
    """Edit message with Markdown, fall back to plain text if parsing fails."""
    try:
        await msg.edit_text(text, parse_mode="Markdown", **kwargs)
    except BadRequest as e:
        if "parse entities" in str(e).lower() or "can't find end" in str(e).lower():
            await msg.edit_text(text, **kwargs)
        else:
            raise


async def safe_reply_text(msg, text, **kwargs):
    """Reply with Markdown, fall back to plain text if parsing fails."""
    try:
        return await msg.reply_text(text, parse_mode="Markdown", **kwargs)
    except BadRequest as e:
        if "parse entities" in str(e).lower() or "can't find end" in str(e).lower():
            return await msg.reply_text(text, **kwargs)
        else:
            raise


async def gemini_with_retry(func, status_msg=None):
    """Call a Gemini function with automatic retry on 503 errors."""
    for attempt in range(1, GEMINI_MAX_RETRIES + 1):
        try:
            return func()
        except Exception as e:
            if "503" in str(e) or "UNAVAILABLE" in str(e).upper():
                if attempt < GEMINI_MAX_RETRIES:
                    logger.warning(
                        f"Gemini 503, retrying ({attempt}/{GEMINI_MAX_RETRIES})..."
                    )
                    if status_msg:
                        await safe_edit_text(
                            status_msg,
                            f"⏳ Gemini is busy, retrying ({attempt}/{GEMINI_MAX_RETRIES})...",
                        )
                    await asyncio.sleep(GEMINI_RETRY_DELAY * attempt)
                else:
                    raise
            else:
                raise


async def download_with_progress(
    pyro_client, chat_id, message_id, tmp_path, status_msg, total_size
):
    """Download via Pyrogram with progress updates."""
    last_percent = -1

    async def progress(current, total):
        nonlocal last_percent
        if total == 0:
            return
        percent = int(current * 100 / total)
        # Only update every 10% to avoid Telegram rate limits
        if percent // 10 > last_percent // 10:
            last_percent = percent
            bar_filled = percent // 10
            bar_empty = 10 - bar_filled
            bar = "█" * bar_filled + "░" * bar_empty
            mb_done = current / (1024 * 1024)
            mb_total = total / (1024 * 1024)
            try:
                await status_msg.edit_text(
                    f"⏬ Downloading: {bar} {percent}%\n"
                    f"{mb_done:.0f}MB / {mb_total:.0f}MB"
                )
            except Exception:
                pass  # ignore rate limit errors

    pyro_msg = await pyro_client.get_messages(chat_id, message_id)
    await pyro_msg.download(file_name=tmp_path, progress=progress)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Hi! I can analyze videos and answer questions about them.\n\n"
        "How to use:\n"
        "1. Send me a video — I'll study it and give you 10 key insights\n"
        "2. Ask me as many questions as you want about it\n"
        "3. Send /done to end the session and analyze a new video\n\n"
        "I auto-extract questions from forwarded messages.\n"
        "Videos of any size are supported!"
    )


async def handle_done(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    session = user_sessions.pop(user_id, None)
    user_last_answer.pop(user_id, None)

    if session:
        try:
            gemini_client.files.delete(name=session["uploaded_file"].name)
        except Exception:
            pass
        await update.message.reply_text(
            "Session ended. Send me a new video to start again."
        )
    else:
        await update.message.reply_text("No active session. Send me a video to start.")


async def handle_copy(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    user_id = query.from_user.id
    answer = user_last_answer.get(user_id)
    if answer:
        await query.message.reply_text(
            f"```\n{answer}\n```",
            parse_mode="Markdown",
        )


async def handle_video(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.message
    user_id = update.effective_user.id

    video = message.video or message.video_note or message.document
    if not video:
        await message.reply_text("Could not read the video. Please try again.")
        return

    # End previous session if exists
    old_session = user_sessions.pop(user_id, None)
    if old_session:
        try:
            gemini_client.files.delete(name=old_session["uploaded_file"].name)
        except Exception:
            pass

    file_size_mb = (video.file_size or 0) / (1024 * 1024)
    status_msg = await message.reply_text(
        f"⏬ Downloading video ({file_size_mb:.0f}MB)..."
    )
    tmp_path = None

    try:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp_path = tmp.name

        # Try Bot API first (<20MB), fall back to Pyrogram with progress for larger
        try:
            file = await context.bot.get_file(video.file_id)
            await file.download_to_drive(custom_path=tmp_path)
        except BadRequest as e:
            if "file is too big" in str(e).lower():
                logger.info("File too big for Bot API, using Pyrogram with progress...")
                await download_with_progress(
                    pyro_client,
                    message.chat_id,
                    message.message_id,
                    tmp_path,
                    status_msg,
                    video.file_size or 0,
                )
            else:
                raise

        actual_size = os.path.getsize(tmp_path) / (1024 * 1024)
        logger.info(f"Downloaded video: {actual_size:.1f} MB")

        await status_msg.edit_text("☁️ Uploading to Gemini...")

        uploaded_file = await gemini_with_retry(
            lambda: gemini_client.files.upload(file=tmp_path),
            status_msg=status_msg,
        )

        while uploaded_file.state.name == "PROCESSING":
            await asyncio.sleep(2)
            uploaded_file = gemini_client.files.get(name=uploaded_file.name)

        if uploaded_file.state.name == "FAILED":
            await status_msg.edit_text(
                "Failed to process the video. Please try a different video."
            )
            return

        await status_msg.edit_text("🔍 Analyzing video and extracting insights...")

        # Create a chat session with the video pre-loaded
        chat = gemini_client.chats.create(
            model=GEMINI_MODEL,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
            ),
            history=[
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_uri(
                            file_uri=uploaded_file.uri,
                            mime_type=uploaded_file.mime_type,
                        ),
                        types.Part.from_text(
                            text="Study this video thoroughly. Confirm you're ready."
                        ),
                    ],
                ),
                types.Content(
                    role="model",
                    parts=[
                        types.Part.from_text(
                            text="I've analyzed the video thoroughly. Ready for your questions."
                        ),
                    ],
                ),
            ],
        )

        # Store session
        user_sessions[user_id] = {
            "uploaded_file": uploaded_file,
            "chat": chat,
        }

        # Generate 10 insights with retry
        insights_response = await gemini_with_retry(
            lambda: chat.send_message(INSIGHTS_PROMPT),
            status_msg=status_msg,
        )

        await safe_edit_text(
            status_msg,
            f"✅ Video analyzed!\n\n"
            f"{insights_response.text}\n\n"
            "—\n"
            "Ask me anything about this video.\n"
            "Send /done when you're finished.",
        )

    except Exception as e:
        logger.exception("Error processing video")
        await status_msg.edit_text(f"Error: {e}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    session = user_sessions.get(user_id)

    if not session:
        await update.message.reply_text(
            "No video loaded. Send me a video first, then ask questions."
        )
        return

    raw_text = update.message.text
    question = extract_question(raw_text)

    status_msg = await update.message.reply_text("Thinking...")

    try:
        response = await gemini_with_retry(
            lambda: session["chat"].send_message(question),
            status_msg=status_msg,
        )
        answer = response.text

        # Store for copy button
        user_last_answer[user_id] = answer

        keyboard = InlineKeyboardMarkup([
            [InlineKeyboardButton("📋 Copy Answer", callback_data="copy_answer")]
        ])

        await safe_edit_text(
            status_msg,
            answer,
            reply_markup=keyboard,
        )
    except Exception as e:
        logger.exception("Error answering question")
        await status_msg.edit_text(f"Error: {e}")


async def main() -> None:
    await pyro_client.start()
    logger.info("Pyrogram client started")

    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("done", handle_done))
    app.add_handler(CallbackQueryHandler(handle_copy, pattern="^copy_answer$"))
    app.add_handler(
        MessageHandler(
            filters.VIDEO | filters.VIDEO_NOTE | filters.Document.VIDEO,
            handle_video,
        )
    )
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    logger.info("Bot started")

    await app.initialize()
    await app.start()
    await app.updater.start_polling()

    try:
        while True:
            await asyncio.sleep(1)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        for session in user_sessions.values():
            try:
                gemini_client.files.delete(name=session["uploaded_file"].name)
            except Exception:
                pass
        user_sessions.clear()

        await app.updater.stop()
        await app.stop()
        await app.shutdown()
        await pyro_client.stop()


if __name__ == "__main__":
    asyncio.run(main())
