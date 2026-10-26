
# app.py
# Streamlit UI for the AI Mock Interview Coach

import base64
import json
import re
import time
from pathlib import Path

import streamlit.components.v1 as components

import streamlit as st
from streamlit_autorefresh import st_autorefresh

import main as core


st.set_page_config(
    page_title="AI Mock Interview Coach",
    page_icon="🎙️",
    layout="wide",
)

st.title("🎙️ AI Mock Interview Coach")
st.caption(
    "Member 1: Resume/JD RAG · "
    "Member 2: Voice Interview · "
    "Member 3: Evaluation and Reports"
)

# ---------------------------------------------------------
# SESSION STATE
# ---------------------------------------------------------

DEFAULTS = {
    "prepared": False,
    "questions": [],
    "role": "Not specified",
    "interview_started": False,
    "interview_finished": False,
    "question_index": 0,
    "deadline": None,
    "question_speech_seconds": 0.0,
    "last_spoken_question": None,
    "answers": [],
    "report": None,
    "session_id": None,
    "recording_key": 0,
    "tts_audio_key": None,
    "tts_audio_bytes": None,
    "tts_error": None,
    "welcome_audio_bytes": None,
    "welcome_audio_error": None,
    "current_followup": None,
    "followup_count": 0,
    "followup_questions": [],
    "followup_answers": [],
}

for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ---------------------------------------------------------
# HELPERS
# ---------------------------------------------------------

def reset_interview():
    st.session_state.interview_started = False
    st.session_state.interview_finished = False
    st.session_state.question_index = 0
    st.session_state.deadline = None
    st.session_state.question_speech_seconds = 0.0
    st.session_state.last_spoken_question = None
    st.session_state.answers = []
    st.session_state.report = None
    st.session_state.session_id = None
    st.session_state.recording_key = 0
    st.session_state.tts_audio_key = None
    st.session_state.tts_audio_bytes = None
    st.session_state.tts_error = None
    st.session_state.welcome_audio_bytes = None
    st.session_state.welcome_audio_error = None
    st.session_state.current_followup = None
    st.session_state.followup_count = 0
    st.session_state.followup_questions = []
    st.session_state.followup_answers = []


def save_uploaded_file(uploaded_file, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(uploaded_file.getvalue())
    return destination


def generate_question_audio(text, question_key):
    """Generate and cache one WAV file per question using main.py's TTS helper."""
    if st.session_state.get("tts_audio_key") == question_key:
        return

    st.session_state.tts_audio_key = question_key
    st.session_state.tts_audio_bytes = None
    st.session_state.tts_error = None

    try:
        core.STORAGE_DIR.mkdir(parents=True, exist_ok=True)
        audio_path = core.STORAGE_DIR / f"question_{question_key}.wav"

        # main.py returns a Path, not audio bytes. Pass the supported output_path.
        generated_path = core.generate_tts_audio(text, output_path=audio_path)
        generated_path = Path(generated_path)
        if not generated_path.exists() or generated_path.stat().st_size == 0:
            raise RuntimeError("The TTS model returned an empty audio file.")

        st.session_state.tts_audio_bytes = generated_path.read_bytes()
    except Exception as exc:
        # Keep the exact error for display and use browser speech as a fallback.
        st.session_state.tts_error = str(exc)


def speak_question_in_browser(text, question_key):
    """Fallback speech that does not consume the Groq TTS quota."""
    import html
    safe_text = json.dumps(str(text))
    safe_key = json.dumps(str(question_key))
    st.html(
        f"""
        <div style="font-family: sans-serif; font-size: 14px;">
          <p>Trying browser speech if audio generation is unavailable…</p>
          <button id="speak-question" style="padding:8px 12px; cursor:pointer;">
            🔊 Read question aloud
          </button>
          <span id="speech-status" style="margin-left:8px;"></span>
        </div>
        <script>
          const questionText = {safe_text};
          const questionKey = {safe_key};
          const button = document.getElementById("speak-question");
          const status = document.getElementById("speech-status");
          function speakQuestion() {{
            if (!("speechSynthesis" in window)) {{
              status.textContent = "Browser speech is not supported. Please read the question and continue.";
              return;
            }}
            window.speechSynthesis.cancel();
            const utterance = new SpeechSynthesisUtterance(questionText);
            utterance.lang = "en-US";
            utterance.rate = 0.9;
            utterance.onstart = () => status.textContent = "Speaking…";
            utterance.onend = () => status.textContent = "Question finished. You can start the timer.";
            utterance.onerror = () => status.textContent = "Speech failed. Please click the button again.";
            window.speechSynthesis.speak(utterance);
          }}
          button.addEventListener("click", speakQuestion);
          // Try automatic speech once for this rendered question; button remains as a fallback.
          speakQuestion();
        </script>
        """,
    )



def generate_welcome_audio():
    'Create and cache the spoken interview welcome message.'
    if st.session_state.get("welcome_audio_bytes"):
        return
    st.session_state.welcome_audio_error = None
    try:
        core.STORAGE_DIR.mkdir(parents=True, exist_ok=True)
        welcome_path = core.STORAGE_DIR / "interview_welcome.wav"
        generated_path = core.generate_tts_audio(
            "Welcome to the AI interview.",
            output_path=welcome_path,
        )
        generated_path = Path(generated_path)
        if not generated_path.exists() or generated_path.stat().st_size == 0:
            raise RuntimeError("Welcome audio was empty.")
        st.session_state.welcome_audio_bytes = generated_path.read_bytes()
    except Exception as exc:
        st.session_state.welcome_audio_error = str(exc)


def render_question_player_and_timer(
    question,
    question_key,
    audio_bytes=None,
    welcome_audio_bytes=None,
    welcome_once_key="ai-interview",
    timer_seconds=60,
):
    """Play the welcome only once per interview, then play this question and start its timer."""
    question_json = json.dumps(str(question))
    welcome_json = json.dumps("Welcome to the AI interview.")
    question_key_json = json.dumps(str(question_key))
    welcome_once_key_json = json.dumps(str(welcome_once_key))

    welcome_markup = ""
    question_markup = ""
    if welcome_audio_bytes:
        welcome_b64 = base64.b64encode(welcome_audio_bytes).decode("ascii")
        welcome_markup = (
            f'<audio id="welcome-audio" preload="auto" '
            f'src="data:audio/wav;base64,{welcome_b64}"></audio>'
        )
    if audio_bytes:
        audio_b64 = base64.b64encode(audio_bytes).decode("ascii")
        question_markup = (
            f'<audio id="question-audio" controls preload="auto" '
            f'src="data:audio/wav;base64,{audio_b64}"></audio>'
        )

    html_doc = f'''<!doctype html>
<html><head><meta charset="utf-8"><style>
body {{font-family:Arial,sans-serif;margin:0;padding:8px;color:#222}}
.timer {{font-size:28px;font-weight:700;margin:10px 0}}
.status {{font-size:14px;margin:6px 0;min-height:20px}}
button {{margin:4px 4px 4px 0;padding:8px 12px;cursor:pointer}}
</style></head><body>
<div><strong>AI Interviewer</strong></div>
<div>{welcome_markup}{question_markup}</div>
<div id="status" class="status">Ready to begin the interview.</div>
<div>Time remaining</div><div id="timer" class="timer">Waiting for the question to finish…</div>
<button id="begin-interview">▶ Begin Welcome and Question</button>
<button id="start-timer" style="display:none">Start Answer Timer</button>
<script>
const questionText = {question_json};
const welcomeText = {welcome_json};
const questionKey = {question_key_json};
const welcomeOnceKey = "aiInterviewWelcomePlayed:" + {welcome_once_key_json};
const limit = {int(timer_seconds)};
const status = document.getElementById('status');
const timer = document.getElementById('timer');
const beginButton = document.getElementById('begin-interview');
const startButton = document.getElementById('start-timer');
const welcomeAudio = document.getElementById('welcome-audio');
const questionAudio = document.getElementById('question-audio');
let started = false;
let sequenceStarted = false;

function startCountdown() {{
  if (started) return;
  started = true;
  startButton.style.display = 'none';
  let remaining = limit;
  timer.textContent = remaining + ' sec';
  status.textContent = 'Your answer time has started. Record your answer and submit it.';
  const interval = setInterval(() => {{
    remaining -= 1;
    timer.textContent = Math.max(remaining, 0) + ' sec';
    if (remaining <= 10 && remaining > 0) status.textContent = 'Time is almost up!';
    if (remaining <= 0) {{
      clearInterval(interval);
      status.textContent = 'Time is up. Submit your recording or skip the question.';
    }}
  }}, 1000);
}}

function speakBrowser(text, onEnd, label) {{
  if (!('speechSynthesis' in window)) {{
    status.textContent = 'Browser speech is unavailable. Read the question and start the timer manually when ready.';
    startButton.style.display = 'inline-block';
    return;
  }}
  const utterance = new SpeechSynthesisUtterance(text);
  utterance.lang = 'en-US';
  utterance.rate = 0.9;
  utterance.onstart = () => status.textContent = label;
  utterance.onend = onEnd;
  utterance.onerror = () => {{
    status.textContent = 'Speech playback failed. Click Begin Welcome and Question to try again.';
    beginButton.style.display = 'inline-block';
  }};
  window.speechSynthesis.speak(utterance);
}}

function playQuestion() {{
  if (questionAudio) {{
    status.textContent = 'AI is asking the interview question…';
    questionAudio.onended = () => {{
      status.textContent = 'Question finished. Starting your answer timer.';
      startCountdown();
    }};
    questionAudio.onerror = () => {{
      status.textContent = 'Question audio failed. Read the question and start the timer when ready.';
      startButton.style.display = 'inline-block';
    }};
    questionAudio.currentTime = 0;
    questionAudio.play().catch(() => {{
      status.textContent = 'Browser blocked audio. Click Play on the question player; the timer starts after it ends.';
      questionAudio.controls = true;
      questionAudio.onended = startCountdown;
    }});
  }} else {{
    speakBrowser(questionText, () => {{
      status.textContent = 'Question finished. Starting your answer timer.';
      startCountdown();
    }}, 'AI is asking the interview question…');
  }}
}}

function hasWelcomeAlreadyPlayed() {{
  try {{ return window.localStorage.getItem(welcomeOnceKey) === '1'; }}
  catch (e) {{ return false; }}
}}
function markWelcomePlayed() {{
  try {{ window.localStorage.setItem(welcomeOnceKey, '1'); }}
  catch (e) {{ /* storage can be unavailable in some browser modes */ }}
}}
function beginSequence() {{
  if (sequenceStarted) return;
  sequenceStarted = true;
  beginButton.style.display = 'none';
  if (window.speechSynthesis) window.speechSynthesis.cancel();

  // Streamlit reruns the component after interactions. Keep the welcome flag in
  // browser storage so it is spoken only once for this interview session.
  if (hasWelcomeAlreadyPlayed()) {{
    playQuestion();
    return;
  }}
  markWelcomePlayed();
  if (welcomeAudio) {{
    status.textContent = 'AI interviewer is welcoming you…';
    welcomeAudio.currentTime = 0;
    welcomeAudio.onended = playQuestion;
    welcomeAudio.onerror = () => speakBrowser(welcomeText, playQuestion, 'AI interviewer is welcoming you…');
    welcomeAudio.play().catch(() => speakBrowser(welcomeText, playQuestion, 'AI interviewer is welcoming you…'));
  }} else {{
    speakBrowser(welcomeText, playQuestion, 'AI interviewer is welcoming you…');
  }}
}}

beginButton.addEventListener('click', beginSequence);
startButton.addEventListener('click', startCountdown);
setTimeout(beginSequence, 300);
</script></body></html>'''
    components.html(html_doc, height=230, scrolling=False)


def start_interview():
    st.session_state.interview_started = True
    st.session_state.interview_finished = False
    st.session_state.question_index = 0
    st.session_state.answers = []
    st.session_state.report = None
    st.session_state.last_spoken_question = None
    # The question phase is accounted for before the answer countdown begins.
    st.session_state.deadline = None
    st.session_state.question_speech_seconds = 0.0
    st.session_state.recording_key = 0
    st.session_state.tts_audio_key = None
    st.session_state.tts_audio_bytes = None
    st.session_state.tts_error = None
    st.session_state.welcome_audio_bytes = None
    st.session_state.welcome_audio_error = None
    st.session_state.current_followup = None
    st.session_state.followup_count = 0
    st.session_state.followup_questions = []
    st.session_state.followup_answers = []

    manager = core.InterviewStateManager()
    st.session_state.session_id = manager.create_session(
        st.session_state.role,
        st.session_state.questions,
    )


def finish_interview(status="completed"):
    """Finish the interview and enrich the final report with follow-ups."""
    st.session_state.interview_finished = True
    st.session_state.interview_started = False

    session_id = st.session_state.session_id
    if session_id:
        manager = core.InterviewStateManager()
        manager.end_interview(session_id, status=status)

    records = st.session_state.answers
    core.save_member2_answers(records, st.session_state.role)

    # Create the standard report first, then append follow-up details.
    report = core.member3_process_saved_interview()
    if isinstance(report, dict):
        follow_up_details = []

        for record in records:
            if not isinstance(record, dict):
                continue

            questions = record.get("follow_up_questions", []) or []
            answers = record.get("follow_up_answers", []) or []
            follow_up_entries = []

            # Match each answer to its actual question, preserving order.
            for index in range(max(len(questions), len(answers))):
                answer_item = (
                    answers[index]
                    if index < len(answers) and isinstance(answers[index], dict)
                    else {}
                )
                question_text = answer_item.get(
                    "question",
                    questions[index] if index < len(questions) else "",
                )
                follow_up_entries.append({
                    "question": question_text,
                    "answer": answer_item.get("answer", ""),
                    "score": answer_item.get("score"),
                    "stt_error": answer_item.get("stt_error"),
                })

            follow_up_details.append({
                "question_number": record.get("question_number"),
                "main_question": record.get("question", ""),
                "main_answer": record.get("answer", ""),
                "main_score": record.get("score"),
                "follow_up_questions": questions,
                "follow_up_answers": follow_up_entries,
            })

        report["follow_up_details"] = follow_up_details
        report["follow_up_summary"] = {
            "total_follow_up_questions": sum(
                len(item["follow_up_questions"])
                for item in follow_up_details
            ),
            "total_follow_up_answers": sum(
                len(item["follow_up_answers"])
                for item in follow_up_details
            ),
        }

        try:
            core.STORAGE_DIR.mkdir(parents=True, exist_ok=True)
            core.FINAL_REPORT_PATH.write_text(
                json.dumps(report, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception as exc:
            st.warning(f"Could not save the complete final report: {exc}")

    st.session_state.report = report


def remaining_seconds():
    deadline = st.session_state.deadline
    if deadline is None:
        return core.ANSWER_TIME_LIMIT_SECONDS
    return max(0, int(deadline - time.time() + 0.999))


def display_score(score):
    if not score:
        st.info("No evaluation score is available for this answer.")
        return

    if score.get("total") is None:
        st.warning(
            "The answer was saved, but evaluation failed. "
            + score.get("evaluation_error", "Unknown evaluation error.")
        )
        return

    st.metric(
        "Answer score",
        f"{score['total']} / {score['max_total']}",
        f"{score['percentage']}%",
    )

    dimensions = score.get("dimensions", {})
    if dimensions:
        cols = st.columns(2)
        for index, (name, item) in enumerate(dimensions.items()):
            with cols[index % 2]:
                st.metric(
                    name.replace("_", " ").title(),
                    f"{item['score']} / 5",
                )
                if item.get("rationale"):
                    st.caption(item["rationale"])


# ---------------------------------------------------------
# SIDEBAR: DOCUMENTS AND PREPARATION
# ---------------------------------------------------------

with st.sidebar:
    st.header("1. Prepare Interview")

    resume_upload = st.file_uploader(
        "Upload resume (PDF or TXT)",
        type=["pdf", "txt"],
        key="resume_upload",
    )

    jd_upload = st.file_uploader(
        "Upload job description (PDF or TXT)",
        type=["pdf", "txt"],
        key="jd_upload",
    )

    if st.button("Generate Interview Questions", use_container_width=True):
        if not resume_upload or not jd_upload:
            st.error("Upload both the resume and job description.")
        else:
            try:
                with st.spinner(
                    "Reading documents, building RAG, and generating questions..."
                ):
                    resume_path = save_uploaded_file(
                        resume_upload,
                        core.DATA_DIR / (
                            "resume" + Path(resume_upload.name).suffix.lower()
                        ),
                    )
                    jd_path = save_uploaded_file(
                        jd_upload,
                        core.DATA_DIR / (
                            "job_description"
                            + Path(jd_upload.name).suffix.lower()
                        ),
                    )

                    questions, role = core.prepare_member1(
                        resume_path,
                        jd_path,
                    )

                    st.session_state.questions = questions
                    st.session_state.role = role
                    st.session_state.prepared = True
                    reset_interview()

                st.success("Interview questions are ready.")

            except Exception as exc:
                st.error(f"Preparation failed: {exc}")

    st.divider()
    st.header("2. Interview Settings")
    st.write(
        f"**Answer time:** "
        f"{core.ANSWER_TIME_LIMIT_SECONDS} seconds per question"
    )
    st.write("**Follow-ups:** Up to 2 per question")
    st.caption(
        "Use headphones if possible. Allow audio playback in your browser."
    )


# ---------------------------------------------------------
# PREPARED QUESTIONS
# ---------------------------------------------------------

if st.session_state.prepared:
    st.subheader("Interview Details")
    st.write(f"**Target role:** {st.session_state.role}")
    st.write(
        f"**Questions prepared:** {len(st.session_state.questions)}"
    )

    # Save generated questions and let the candidate download them.
    questions_payload = {
        "role": st.session_state.role,
        "questions": st.session_state.questions,
    }
    questions_json = json.dumps(
        questions_payload, indent=2, ensure_ascii=False
    )
    try:
        core.STORAGE_DIR.mkdir(parents=True, exist_ok=True)
        core.QUESTIONS_PATH.write_text(questions_json, encoding="utf-8")
    except Exception as exc:
        st.warning(f"Could not save the questions JSON: {exc}")

    st.download_button(
        "⬇️ Download Questions JSON",
        data=questions_json.encode("utf-8"),
        file_name="interview_questions.json",
        mime="application/json",
        key="download_interview_questions",
    )

    stage_names_preview = [
        "Stage 1 — Introduction",
        "Stage 2 — Resume and Projects",
        "Stage 3 — Technical Skills",
        "Stage 4 — Problem-Solving and Challenges",
        "Stage 5 — Job Requirements",
        "Stage 6 — Programming and Reasoning",
        "Stage 7 — Role Fit",
    ]
    with st.expander("👁️ View / Hide Generated Questions", expanded=False):
        st.markdown("#### Generated interview questions")
        for i, generated_question in enumerate(st.session_state.questions):
            stage_name = stage_names_preview[min(i, len(stage_names_preview) - 1)]
            st.markdown(f"**{stage_name}**")
            st.write(f"{i + 1}. {generated_question}")

    if not st.session_state.interview_started and not st.session_state.interview_finished:
        if st.button("Start Interview", type="primary"):
            try:
                start_interview()
                st.rerun()
            except Exception as exc:
                st.error(f"Could not start interview: {exc}")


# ---------------------------------------------------------
# ACTIVE INTERVIEW
# ---------------------------------------------------------

if st.session_state.interview_started:
    index = st.session_state.question_index
    questions = st.session_state.questions

    if index >= len(questions):
        finish_interview("completed")
        st.rerun()

    # Do not force periodic Streamlit reruns here: they can race with the
    # candidate clicking Submit Answer and make the click appear to be ignored.
    # The countdown refreshes on the next user interaction.

    main_question = questions[index]
    is_followup = bool(st.session_state.current_followup)
    question = st.session_state.current_followup if is_followup else main_question
    question_key = f"{st.session_state.session_id}_{index}_followup_{st.session_state.followup_count}" if is_followup else f"{st.session_state.session_id}_{index}"

    st.divider()
    stage_names = [
        "Stage 1: Introduction",
        "Stage 2: Resume and Projects",
        "Stage 3: Technical Skills",
        "Stage 4: Problem-Solving and Challenges",
        "Stage 5: Job Requirements",
        "Stage 6: Programming and Reasoning",
        "Stage 7: Role Fit",
    ]
    if is_followup:
        st.subheader(f"Follow-up {st.session_state.followup_count} of {core.MAX_FOLLOW_UPS}")
        st.caption(f"Follow-up for main question {index + 1}: {stage_names[min(index, len(stage_names) - 1)]}")
    else:
        stage_label = stage_names[min(index, len(stage_names) - 1)]
        st.subheader(f"{stage_label} · Question {index + 1} of {len(questions)}")
        st.progress((index + 1) / len(questions))
    st.markdown(f"### {question}")

    # Generate question audio, then show a live browser timer that starts only
    # when the actual audio/TTS playback ends.
    generate_question_audio(question, question_key)

    if st.session_state.tts_error:
        st.warning(
            "Groq TTS could not generate audio; browser speech will be used. "
            "Details: " + st.session_state.tts_error
        )

    # Speak the welcome message before the first main question only.
    if index == 0 and not is_followup:
        generate_welcome_audio()
        if st.session_state.welcome_audio_error:
            st.caption("Welcome audio will use browser speech fallback: " + st.session_state.welcome_audio_error)

    render_question_player_and_timer(
        question=question,
        question_key=question_key,
        audio_bytes=st.session_state.tts_audio_bytes,
        welcome_audio_bytes=(
            st.session_state.welcome_audio_bytes
            if index == 0 and not is_followup
            else None
        ),
        welcome_once_key=st.session_state.session_id or "ai-interview",
        timer_seconds=core.ANSWER_TIME_LIMIT_SECONDS,
    )

    st.caption(
        "Record your answer using the microphone control, then click Submit Answer. "
        "The live countdown is shown above and starts after the question finishes playing."
    )

    # Streamlit's audio input requires the candidate to control recording.
    audio = st.audio_input(
        "Record your answer",
        key=f"answer_audio_{question_key}_{st.session_state.recording_key}",
    )

    submit_col, skip_col = st.columns(2)

    with submit_col:
        # Do not disable submission because of the timer state. If a recording
        # exists, the candidate must be able to submit it and move forward.
        submit_answer = st.button(
            "Submit Answer",
            type="primary",
            disabled=(audio is None),
            use_container_width=True,
            key=f"submit_answer_{question_key}_{st.session_state.recording_key}",
        )

    with skip_col:
        skip_question = st.button(
            "Skip Question",
            use_container_width=True,
        )

    if submit_answer:
        audio_bytes = b""
        transcript = ""
        stt_error = None
        recording_path = None
        score = {
            "question": question,
            "answer": "",
            "dimensions": {},
            "total": None,
            "max_total": 20,
            "percentage": None,
            "evaluation_error": "Answer was not scored.",
        }

        if audio is not None:
            try:
                audio_bytes = audio.getvalue() if hasattr(audio, "getvalue") else bytes(audio)
            except Exception as exc:
                stt_error = f"Could not read the recorded audio: {exc}"

        if not audio_bytes:
            st.error("No recording was received. Record your answer again, then click Submit Answer.")
        else:
            with st.spinner("Processing answer: saving audio, transcribing, and evaluating..."):
                core.STORAGE_DIR.mkdir(parents=True, exist_ok=True)
                safe_question_key = re.sub(r"[^a-zA-Z0-9_-]", "_", str(question_key))
                recording_path = core.STORAGE_DIR / f"answer_{safe_question_key}.wav"
                try:
                    recording_path.write_bytes(audio_bytes)
                except Exception as exc:
                    stt_error = f"Audio file could not be saved: {exc}"

                try:
                    transcript = (core.transcribe_audio(audio_bytes, filename=recording_path.name) or "").strip()
                    if not transcript:
                        stt_error = "STT returned an empty transcript. The recording was retained if saving succeeded."
                except Exception as exc:
                    stt_error = f"STT failed: {exc}"

                if transcript:
                    try:
                        score = core.score_answer(question, transcript)
                    except Exception as exc:
                        score["answer"] = transcript
                        score["evaluation_error"] = str(exc)

                if is_followup:
                    # Store follow-up answers under the original main question.
                    parent_record = next(
                        (r for r in reversed(st.session_state.answers)
                         if r.get("question") == main_question),
                        None,
                    )
                    if parent_record is None:
                        parent_record = {
                            "question_number": index + 1,
                            "stage": "Interview",
                            "category": "resume_or_job_description",
                            "question": main_question,
                            "answer": "",
                            "audio_file": None,
                            "stt_error": None,
                            "follow_up_questions": list(st.session_state.followup_questions),
                            "follow_up_answers": [],
                            "score": None,
                            "started_at": None,
                            "ended_at": core.utc_now(),
                        }
                        st.session_state.answers.append(parent_record)
                    parent_record.setdefault("follow_up_answers", []).append({
                        "question": question,
                        "answer": transcript,
                        "audio_file": str(recording_path) if recording_path else None,
                        "score": score,
                        "stt_error": stt_error,
                    })
                    previous_followups = list(parent_record.get("follow_up_questions", []))
                    if transcript and len(previous_followups) < core.MAX_FOLLOW_UPS:
                        try:
                            next_followup = core.decide_follow_up(
                                main_question,
                                transcript,
                                previous_followups=previous_followups,
                                score=score,
                            )
                        except Exception as exc:
                            next_followup = ""
                            st.warning(f"Could not generate another follow-up: {exc}")
                    else:
                        next_followup = ""

                    if next_followup and len(previous_followups) < core.MAX_FOLLOW_UPS:
                        parent_record["follow_up_questions"].append(next_followup)
                        st.session_state.followup_questions = list(parent_record["follow_up_questions"])
                        st.session_state.current_followup = next_followup
                        st.session_state.followup_count = len(parent_record["follow_up_questions"])
                    else:
                        st.session_state.current_followup = None
                        st.session_state.followup_count = 0
                        st.session_state.followup_questions = []
                        st.session_state.followup_answers = []
                        st.session_state.question_index = index + 1
                else:
                    # Decide on a follow-up after scoring the main answer.
                    try:
                        first_followup = core.decide_follow_up(
                            main_question,
                            transcript,
                            previous_followups=[],
                            score=score,
                        ) if transcript else ""
                    except Exception as exc:
                        first_followup = ""
                        st.warning(f"Could not generate a follow-up: {exc}")

                    record = {
                        "question_number": index + 1,
                        "stage": "Interview",
                        "category": "resume_or_job_description",
                        "question": main_question,
                        "answer": transcript,
                        "audio_file": str(recording_path) if recording_path else None,
                        "stt_error": stt_error,
                        "follow_up_questions": [first_followup] if first_followup else [],
                        "follow_up_answers": [],
                        "score": score,
                        "started_at": None,
                        "ended_at": core.utc_now(),
                    }
                    st.session_state.answers.append(record)

                    if first_followup:
                        st.session_state.current_followup = first_followup
                        st.session_state.followup_count = 1
                        st.session_state.followup_questions = [first_followup]
                        st.session_state.followup_answers = []
                    else:
                        st.session_state.current_followup = None
                        st.session_state.followup_count = 0
                        st.session_state.followup_questions = []
                        st.session_state.followup_answers = []
                        st.session_state.question_index = index + 1

                # Save the main answer and its follow-up thread to SQLite/JSON.
                parent_record = next(
                    (r for r in reversed(st.session_state.answers)
                     if r.get("question") == main_question),
                    None,
                )
                if parent_record:
                    try:
                        manager = core.InterviewStateManager()
                        manager.save_answer(
                            st.session_state.session_id,
                            index + 1,
                            parent_record.get("answer", ""),
                            parent_record.get("follow_up_questions", []),
                            parent_record.get("follow_up_answers", []),
                            parent_record.get("score"),
                        )
                    except Exception as exc:
                        st.warning(f"SQLite save failed: {exc}")

                try:
                    core.save_member2_answers(st.session_state.answers, st.session_state.role)
                except Exception as exc:
                    st.warning(f"JSON save failed: {exc}")

            # Reset playback/timer for the next prompt (main or follow-up).
            st.session_state.deadline = None
            st.session_state.question_speech_seconds = 0.0
            st.session_state.recording_key += 1
            st.session_state.last_spoken_question = None
            st.session_state.tts_audio_key = None
            st.session_state.tts_audio_bytes = None
            st.session_state.tts_error = None

            if not st.session_state.current_followup and st.session_state.question_index >= len(questions):
                try:
                    finish_interview("completed")
                except Exception as exc:
                    st.error(f"Answers were processed, but final report generation failed: {exc}")
                    st.session_state.interview_finished = True
                    st.session_state.interview_started = False

            if transcript:
                if st.session_state.current_followup:
                    st.success("Answer saved. The AI will ask a follow-up question next.")
                else:
                    st.success("Answer processed. Moving to the next main question.")
                st.write("**Recognized answer:**", transcript)
                try:
                    display_score(score)
                except Exception as exc:
                    st.warning(f"Score display failed, but the answer was processed: {exc}")
            else:
                st.warning("The recording was submitted, but STT did not return text. The interview will continue.")
                if stt_error:
                    st.caption(stt_error)
            st.rerun()

    if skip_question:
        if is_followup:
            # Skipping a follow-up ends the follow-up chain, not the main question list.
            st.session_state.current_followup = None
            st.session_state.followup_count = 0
            st.session_state.followup_questions = []
            st.session_state.followup_answers = []
            st.session_state.question_index = index + 1
        else:
            empty_score = core.score_answer(main_question, "")
            record = {
                "question_number": index + 1,
                "stage": "Interview",
                "category": "resume_or_job_description",
                "question": main_question,
                "answer": "",
                "follow_up_questions": [],
                "follow_up_answers": [],
                "score": empty_score,
                "started_at": None,
                "ended_at": core.utc_now(),
            }
            st.session_state.answers.append(record)
            try:
                manager = core.InterviewStateManager()
                manager.save_answer(st.session_state.session_id, index + 1, "", [], [], empty_score)
            except Exception as exc:
                st.warning(f"Could not update SQLite for skipped question: {exc}")
            st.session_state.question_index = index + 1

        core.save_member2_answers(st.session_state.answers, st.session_state.role)
        st.session_state.deadline = None
        st.session_state.question_speech_seconds = 0.0
        st.session_state.recording_key += 1
        st.session_state.last_spoken_question = None
        st.session_state.tts_audio_key = None
        st.session_state.tts_audio_bytes = None
        st.session_state.tts_error = None

        if st.session_state.question_index >= len(questions):
            finish_interview("completed")
        st.rerun()

    if st.button("End Interview Early"):
        finish_interview("interrupted")
        st.rerun()


# ---------------------------------------------------------
# FINAL REPORT AND SCORES
# ---------------------------------------------------------

if st.session_state.interview_finished:
    st.divider()
    st.header("📊 Interview Evaluation Report")

    report = st.session_state.report

    if report:
        summary = report.get("score_summary", {})
        overall = summary.get("overall_percentage")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Questions answered",
                report.get("questions_answered", 0),
            )

        with col2:
            st.metric(
                "Answers evaluated",
                summary.get("scored_answers", 0),
            )

        with col3:
            st.metric(
                "Overall score",
                f"{overall}%" if overall is not None else "Unavailable",
            )

        st.subheader("Individual Answer Scores")

        for number, score in enumerate(
            report.get("answer_scores", []), start=1
        ):
            with st.expander(
                f"Question {number}: "
                f"{score.get('total', 'N/A')} / "
                f"{score.get('max_total', 20)}"
            ):
                st.write("**Question:**", score.get("question", ""))
                st.write("**Answer:**", score.get("answer", ""))
                display_score(score)

        st.subheader("🔄 Follow-up Questions and Answers")
        follow_summary = report.get("follow_up_summary", {})
        follow_col1, follow_col2 = st.columns(2)
        with follow_col1:
            st.metric(
                "Follow-up questions",
                follow_summary.get("total_follow_up_questions", 0),
            )
        with follow_col2:
            st.metric(
                "Follow-up answers",
                follow_summary.get("total_follow_up_answers", 0),
            )

        follow_details = report.get("follow_up_details", [])
        visible_followups = [
            item for item in follow_details
            if item.get("follow_up_questions") or item.get("follow_up_answers")
        ]
        if not visible_followups:
            st.info("No follow-up questions were asked during this interview.")
        else:
            for item in visible_followups:
                with st.expander(
                    f"Main Question {item.get('question_number', '')}: "
                    f"{item.get('main_question', '')}"
                ):
                    st.markdown("**Main answer:**")
                    st.write(item.get("main_answer", "") or "No answer recorded.")
                    if item.get("main_score"):
                        st.markdown("**Main answer evaluation:**")
                        display_score(item["main_score"])

                    questions = item.get("follow_up_questions", []) or []
                    answers = item.get("follow_up_answers", []) or []
                    for index in range(max(len(questions), len(answers))):
                        answer_item = answers[index] if index < len(answers) else {}
                        question_text = answer_item.get(
                            "question",
                            questions[index] if index < len(questions) else "",
                        )
                        st.markdown(f"**Follow-up {index + 1}:**")
                        st.write(question_text or "Question not recorded.")
                        if index < len(answers):
                            st.markdown("**Candidate's answer:**")
                            st.write(answer_item.get("answer", "") or "No answer recorded.")
                            if answer_item.get("stt_error"):
                                st.caption(answer_item["stt_error"])
                            if answer_item.get("score"):
                                st.markdown("**Follow-up evaluation:**")
                                display_score(answer_item["score"])
                        else:
                            st.info("No answer recorded for this follow-up.")

        st.subheader("Downloadable Report")
        st.json(report)

        report_bytes = json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ).encode("utf-8")

        st.download_button(
            "Download JSON Report",
            data=report_bytes,
            file_name="final_report.json",
            mime="application/json",
        )

    else:
        st.warning(
            "The interview ended, but the final report could not be created. "
            "Check the saved answers and the Groq API configuration."
        )

    if st.button("Start a New Interview"):
        reset_interview()
        st.session_state.prepared = False
        st.session_state.questions = []
        st.rerun()

elif not st.session_state.prepared:
    st.info(
        "Upload your resume and job description in the sidebar, "
        "then generate your interview questions."
    )
