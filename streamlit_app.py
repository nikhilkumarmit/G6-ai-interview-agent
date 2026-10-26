
# # app.py
# # Streamlit UI for the AI Mock Interview Coach

# import json
# import time
# from pathlib import Path

# import streamlit as st
# from streamlit_autorefresh import st_autorefresh

# import main as core


# st.set_page_config(
#     page_title="AI Mock Interview Coach",
#     page_icon="🎙️",
#     layout="wide",
# )

# st.title("🎙️ AI Mock Interview Coach")
# st.caption(
#     "Member 1: Resume/JD RAG · "
#     "Member 2: Voice Interview · "
#     "Member 3: Evaluation and Reports"
# )

# # ---------------------------------------------------------
# # SESSION STATE
# # ---------------------------------------------------------

# DEFAULTS = {
#     "prepared": False,
#     "questions": [],
#     "role": "Not specified",
#     "interview_started": False,
#     "interview_finished": False,
#     "question_index": 0,
#     "deadline": None,
#     "question_speech_seconds": 0.0,
#     "last_spoken_question": None,
#     "answers": [],
#     "report": None,
#     "session_id": None,
#     "recording_key": 0,
#     "tts_audio_key": None,
#     "tts_audio_bytes": None,
#     "tts_error": None,
# }

# for key, value in DEFAULTS.items():
#     if key not in st.session_state:
#         st.session_state[key] = value


# # ---------------------------------------------------------
# # HELPERS
# # ---------------------------------------------------------

# def reset_interview():
#     st.session_state.interview_started = False
#     st.session_state.interview_finished = False
#     st.session_state.question_index = 0
#     st.session_state.deadline = None
#     st.session_state.question_speech_seconds = 0.0
#     st.session_state.last_spoken_question = None
#     st.session_state.answers = []
#     st.session_state.report = None
#     st.session_state.session_id = None
#     st.session_state.recording_key = 0
#     st.session_state.tts_audio_key = None
#     st.session_state.tts_audio_bytes = None
#     st.session_state.tts_error = None


# def save_uploaded_file(uploaded_file, destination):
#     destination = Path(destination)
#     destination.parent.mkdir(parents=True, exist_ok=True)
#     destination.write_bytes(uploaded_file.getvalue())
#     return destination


# def generate_question_audio(text, question_key):
#     """Generate and cache one WAV file per question using main.py's TTS helper."""
#     if st.session_state.get("tts_audio_key") == question_key:
#         return

#     st.session_state.tts_audio_key = question_key
#     st.session_state.tts_audio_bytes = None
#     st.session_state.tts_error = None

#     try:
#         core.STORAGE_DIR.mkdir(parents=True, exist_ok=True)
#         audio_path = core.STORAGE_DIR / f"question_{question_key}.wav"

#         # main.py returns a Path, not audio bytes. Pass the supported output_path.
#         generated_path = core.generate_tts_audio(text, output_path=audio_path)
#         generated_path = Path(generated_path)
#         if not generated_path.exists() or generated_path.stat().st_size == 0:
#             raise RuntimeError("The TTS model returned an empty audio file.")

#         st.session_state.tts_audio_bytes = generated_path.read_bytes()
#     except Exception as exc:
#         # Keep the exact error for display and use browser speech as a fallback.
#         st.session_state.tts_error = str(exc)


# def speak_question_in_browser(text, question_key):
#     """Fallback speech that does not consume the Groq TTS quota."""
#     import html
#     safe_text = json.dumps(str(text))
#     safe_key = json.dumps(str(question_key))
#     st.html(
#         f"""
#         <div style="font-family: sans-serif; font-size: 14px;">
#           <p>Trying browser speech if audio generation is unavailable…</p>
#           <button id="speak-question" style="padding:8px 12px; cursor:pointer;">
#             🔊 Read question aloud
#           </button>
#           <span id="speech-status" style="margin-left:8px;"></span>
#         </div>
#         <script>
#           const questionText = {safe_text};
#           const questionKey = {safe_key};
#           const button = document.getElementById("speak-question");
#           const status = document.getElementById("speech-status");
#           function speakQuestion() {{
#             if (!("speechSynthesis" in window)) {{
#               status.textContent = "Browser speech is not supported. Please read the question and continue.";
#               return;
#             }}
#             window.speechSynthesis.cancel();
#             const utterance = new SpeechSynthesisUtterance(questionText);
#             utterance.lang = "en-US";
#             utterance.rate = 0.9;
#             utterance.onstart = () => status.textContent = "Speaking…";
#             utterance.onend = () => status.textContent = "Question finished. You can start the timer.";
#             utterance.onerror = () => status.textContent = "Speech failed. Please click the button again.";
#             window.speechSynthesis.speak(utterance);
#           }}
#           button.addEventListener("click", speakQuestion);
#           // Try automatic speech once for this rendered question; button remains as a fallback.
#           speakQuestion();
#         </script>
#         """,
#     )


# def start_interview():
#     st.session_state.interview_started = True
#     st.session_state.interview_finished = False
#     st.session_state.question_index = 0
#     st.session_state.answers = []
#     st.session_state.report = None
#     st.session_state.last_spoken_question = None
#     # The question phase is accounted for before the answer countdown begins.
#     st.session_state.deadline = None
#     st.session_state.question_speech_seconds = 0.0
#     st.session_state.recording_key = 0
#     st.session_state.tts_audio_key = None
#     st.session_state.tts_audio_bytes = None
#     st.session_state.tts_error = None

#     manager = core.InterviewStateManager()
#     st.session_state.session_id = manager.create_session(
#         st.session_state.role,
#         st.session_state.questions,
#     )


# def finish_interview(status="completed"):
#     st.session_state.interview_finished = True
#     st.session_state.interview_started = False

#     session_id = st.session_state.session_id
#     if session_id:
#         manager = core.InterviewStateManager()
#         manager.end_interview(session_id, status=status)

#     records = st.session_state.answers
#     core.save_member2_answers(records, st.session_state.role)

#     # Create final report from saved answers and show it in this app.
#     report = core.member3_process_saved_interview()
#     st.session_state.report = report


# def remaining_seconds():
#     deadline = st.session_state.deadline
#     if deadline is None:
#         return core.ANSWER_TIME_LIMIT_SECONDS
#     return max(0, int(deadline - time.time() + 0.999))


# def display_score(score):
#     if not score:
#         st.info("No evaluation score is available for this answer.")
#         return

#     if score.get("total") is None:
#         st.warning(
#             "The answer was saved, but evaluation failed. "
#             + score.get("evaluation_error", "Unknown evaluation error.")
#         )
#         return

#     st.metric(
#         "Answer score",
#         f"{score['total']} / {score['max_total']}",
#         f"{score['percentage']}%",
#     )

#     dimensions = score.get("dimensions", {})
#     if dimensions:
#         cols = st.columns(2)
#         for index, (name, item) in enumerate(dimensions.items()):
#             with cols[index % 2]:
#                 st.metric(
#                     name.replace("_", " ").title(),
#                     f"{item['score']} / 5",
#                 )
#                 if item.get("rationale"):
#                     st.caption(item["rationale"])


# # ---------------------------------------------------------
# # SIDEBAR: DOCUMENTS AND PREPARATION
# # ---------------------------------------------------------

# with st.sidebar:
#     st.header("1. Prepare Interview")

#     resume_upload = st.file_uploader(
#         "Upload resume (PDF or TXT)",
#         type=["pdf", "txt"],
#         key="resume_upload",
#     )

#     jd_upload = st.file_uploader(
#         "Upload job description (PDF or TXT)",
#         type=["pdf", "txt"],
#         key="jd_upload",
#     )

#     if st.button("Generate Interview Questions", use_container_width=True):
#         if not resume_upload or not jd_upload:
#             st.error("Upload both the resume and job description.")
#         else:
#             try:
#                 with st.spinner(
#                     "Reading documents, building RAG, and generating questions..."
#                 ):
#                     resume_path = save_uploaded_file(
#                         resume_upload,
#                         core.DATA_DIR / (
#                             "resume" + Path(resume_upload.name).suffix.lower()
#                         ),
#                     )
#                     jd_path = save_uploaded_file(
#                         jd_upload,
#                         core.DATA_DIR / (
#                             "job_description"
#                             + Path(jd_upload.name).suffix.lower()
#                         ),
#                     )

#                     questions, role = core.prepare_member1(
#                         resume_path,
#                         jd_path,
#                     )

#                     st.session_state.questions = questions
#                     st.session_state.role = role
#                     st.session_state.prepared = True
#                     reset_interview()

#                 st.success("Interview questions are ready.")

#             except Exception as exc:
#                 st.error(f"Preparation failed: {exc}")

#     st.divider()
#     st.header("2. Interview Settings")
#     st.write(
#         f"**Answer time:** "
#         f"{core.ANSWER_TIME_LIMIT_SECONDS} seconds per question"
#     )
#     st.write("**Follow-ups:** Up to 2 per question")
#     st.caption(
#         "Use headphones if possible. Allow audio playback in your browser."
#     )


# # ---------------------------------------------------------
# # PREPARED QUESTIONS
# # ---------------------------------------------------------

# if st.session_state.prepared:
#     st.subheader("Interview Details")
#     st.write(f"**Target role:** {st.session_state.role}")
#     st.write(
#         f"**Questions prepared:** {len(st.session_state.questions)}"
#     )

#     with st.expander("View generated questions"):
#         for i, question in enumerate(
#             st.session_state.questions, start=1
#         ):
#             st.write(f"{i}. {question}")

#     if not st.session_state.interview_started and not st.session_state.interview_finished:
#         if st.button("Start Interview", type="primary"):
#             try:
#                 start_interview()
#                 st.rerun()
#             except Exception as exc:
#                 st.error(f"Could not start interview: {exc}")


# # ---------------------------------------------------------
# # ACTIVE INTERVIEW
# # ---------------------------------------------------------

# if st.session_state.interview_started:
#     index = st.session_state.question_index
#     questions = st.session_state.questions

#     if index >= len(questions):
#         finish_interview("completed")
#         st.rerun()

#     # Do not force periodic Streamlit reruns here: they can race with the
#     # candidate clicking Submit Answer and make the click appear to be ignored.
#     # The countdown refreshes on the next user interaction.

#     question = questions[index]
#     question_key = f"{st.session_state.session_id}_{index}"

#     st.divider()
#     st.subheader(
#         f"Question {index + 1} of {len(questions)}"
#     )
#     st.progress((index + 1) / len(questions))
#     st.markdown(f"### {question}")

#     # Generate a real audio file for each question and show the player.
#     generate_question_audio(question, question_key)

#     # Estimate spoken-question duration. The answer countdown is scheduled
#     # to begin after the question should have finished being read aloud.
#     # Browser speech is used when Groq TTS is unavailable.
#     words = max(1, len(question.split()))
#     estimated_speech_seconds = max(2.0, words / 2.4)
#     if st.session_state.tts_audio_bytes:
#         st.markdown("### 🔊 Listen to the interview question")
#         st.audio(st.session_state.tts_audio_bytes, format="audio/wav")
#         st.caption("The answer timer starts automatically after the question's estimated speaking time. You can also press Play if your browser does not autoplay audio.")
#     else:
#         if st.session_state.tts_error:
#             st.warning(
#                 "Groq TTS could not generate audio. Browser speech is available as a fallback. "
#                 "Details: " + st.session_state.tts_error
#             )
#         else:
#             st.info("Preparing question audio…")
#         if st.session_state.last_spoken_question != question_key:
#             st.session_state.last_spoken_question = question_key
#             speak_question_in_browser(question, question_key)
#         st.caption("The answer timer starts automatically after the estimated question speaking time. If browser speech does not start, click Read question aloud.")

#     # Start the timer automatically for this question. Add the estimated
#     # question-reading duration so the full answer window remains available.
#     if st.session_state.deadline is None:
#         st.session_state.question_speech_seconds = estimated_speech_seconds
#         st.session_state.deadline = (
#             time.time() + estimated_speech_seconds + core.ANSWER_TIME_LIMIT_SECONDS
#         )

#     timer_started = time.time() >= (
#         st.session_state.deadline - core.ANSWER_TIME_LIMIT_SECONDS
#     )
#     seconds = max(
#         0,
#         min(
#             core.ANSWER_TIME_LIMIT_SECONDS,
#             int(st.session_state.deadline - time.time() + 0.999),
#         ),
#     )
#     # Keep the full answer-time display during the question's speaking phase.
#     if not timer_started:
#         seconds = core.ANSWER_TIME_LIMIT_SECONDS

#     timer_col, status_col = st.columns([1, 2])
#     with timer_col:
#         st.metric(
#             "Time remaining",
#             f"{seconds} sec" if timer_started else "Starts after question",
#         )
#     with status_col:
#         if not timer_started:
#             st.info("Listen to the question. Your answer timer will start automatically.")
#         elif seconds > 10:
#             st.info("Record your answer and submit before the timer ends.")
#         elif seconds > 0:
#             st.warning("Time is almost up.")
#         else:
#             st.error("Time expired. This answer can no longer be submitted. Move to the next question.")

#     st.caption(
#         "Click the microphone control to record your answer, then "
#         "submit it. The timer refreshes automatically."
#     )

#     # Streamlit's audio input requires the candidate to control recording.
#     audio = st.audio_input(
#         "Record your answer",
#         key=f"answer_audio_{question_key}_{st.session_state.recording_key}",
#     )

#     submit_col, skip_col = st.columns(2)

#     with submit_col:
#         # Do not disable submission because of the timer state. If a recording
#         # exists, the candidate must be able to submit it and move forward.
#         submit_answer = st.button(
#             "Submit Answer",
#             type="primary",
#             disabled=(audio is None),
#             use_container_width=True,
#             key=f"submit_answer_{question_key}_{st.session_state.recording_key}",
#         )

#     with skip_col:
#         skip_question = st.button(
#             "Skip Question",
#             use_container_width=True,
#         )

#     if submit_answer:
#         # The recording has been received. Always attempt processing, then advance
#         # the question even if STT, scoring, or a persistence service fails.
#         audio_bytes = b""
#         transcript = ""
#         stt_error = None
#         recording_path = None
#         score = {
#             "question": question,
#             "answer": "",
#             "dimensions": {},
#             "total": None,
#             "max_total": 20,
#             "percentage": None,
#             "evaluation_error": "Answer was not scored.",
#         }
#         if audio is not None:
#             try:
#                 audio_bytes = audio.getvalue() if hasattr(audio, "getvalue") else bytes(audio)
#             except Exception as exc:
#                 stt_error = f"Could not read the recorded audio: {exc}"

#         if not audio_bytes:
#             st.error("No recording was received. Record your answer again, then click Submit Answer.")
#         else:
#             with st.spinner("Processing answer: saving audio, transcribing, and recording results..."):
#                 core.STORAGE_DIR.mkdir(parents=True, exist_ok=True)
#                 recording_path = core.STORAGE_DIR / f"answer_{st.session_state.session_id}_{index + 1}.wav"
#                 try:
#                     recording_path.write_bytes(audio_bytes)
#                 except Exception as exc:
#                     stt_error = f"Audio file could not be saved: {exc}"

#                 # STT
#                 try:
#                     transcript = (core.transcribe_audio(audio_bytes, filename=recording_path.name) or "").strip()
#                     if not transcript:
#                         stt_error = "STT returned an empty transcript. The recording was retained if saving succeeded."
#                 except Exception as exc:
#                     stt_error = f"STT failed: {exc}"

#                 # Score only when a transcript exists.
#                 if transcript:
#                     try:
#                         score = core.score_answer(question, transcript)
#                     except Exception as exc:
#                         score["answer"] = transcript
#                         score["evaluation_error"] = str(exc)

#                 record = {
#                     "question_number": index + 1,
#                     "stage": "Interview",
#                     "category": "resume_or_job_description",
#                     "question": question,
#                     "answer": transcript,
#                     "audio_file": str(recording_path) if recording_path else None,
#                     "stt_error": stt_error,
#                     "follow_up_questions": [],
#                     "follow_up_answers": [],
#                     "score": score,
#                     "started_at": None,
#                     "ended_at": core.utc_now(),
#                 }
#                 st.session_state.answers.append(record)

#                 try:
#                     manager = core.InterviewStateManager()
#                     manager.save_answer(
#                         st.session_state.session_id,
#                         index + 1,
#                         transcript or "[STT failed; audio saved if possible]",
#                         [], [], score,
#                     )
#                 except Exception as exc:
#                     st.warning(f"SQLite save failed: {exc}")

#                 try:
#                     core.save_member2_answers(st.session_state.answers, st.session_state.role)
#                 except Exception as exc:
#                     st.warning(f"JSON save failed: {exc}")

#             # IMPORTANT: advance state before rendering optional score/details.
#             # This ensures a UI/reporting issue cannot leave the same question visible.
#             st.session_state.question_index = index + 1
#             st.session_state.deadline = None
#             st.session_state.question_speech_seconds = 0.0
#             st.session_state.recording_key += 1
#             st.session_state.last_spoken_question = None
#             st.session_state.tts_audio_key = None
#             st.session_state.tts_audio_bytes = None
#             st.session_state.tts_error = None

#             if st.session_state.question_index >= len(questions):
#                 try:
#                     finish_interview("completed")
#                 except Exception as exc:
#                     st.error(f"Answers were processed, but final report generation failed: {exc}")
#                     st.session_state.interview_finished = True
#                     st.session_state.interview_started = False

#             if transcript:
#                 st.success("Answer processed. Moving to the next question.")
#                 st.write("**Recognized answer:**", transcript)
#                 try:
#                     display_score(score)
#                 except Exception as exc:
#                     st.warning(f"Score display failed, but the answer was processed: {exc}")
#             else:
#                 st.warning("The recording was submitted, but STT did not return text. The question has still been advanced.")
#                 if stt_error:
#                     st.caption(stt_error)
#             st.rerun()

#     if skip_question:
#         # Save a zero-score record for the skipped question.
#         empty_score = core.score_answer(question, "")
#         record = {
#             "question_number": index + 1,
#             "stage": "Interview",
#             "category": "resume_or_job_description",
#             "question": question,
#             "answer": "",
#             "follow_up_questions": [],
#             "follow_up_answers": [],
#             "score": empty_score,
#             "started_at": None,
#             "ended_at": core.utc_now(),
#         }

#         manager = core.InterviewStateManager()
#         manager.save_answer(
#             st.session_state.session_id,
#             index + 1,
#             "",
#             [],
#             [],
#             empty_score,
#         )

#         st.session_state.answers.append(record)
#         core.save_member2_answers(
#             st.session_state.answers,
#             st.session_state.role,
#         )

#         st.session_state.question_index += 1
#         st.session_state.deadline = None
#         st.session_state.question_speech_seconds = 0.0
#         st.session_state.recording_key += 1
#         st.session_state.last_spoken_question = None

#         if st.session_state.question_index >= len(questions):
#             finish_interview("completed")

#         st.rerun()

#     if st.button("End Interview Early"):
#         finish_interview("interrupted")
#         st.rerun()


# # ---------------------------------------------------------
# # FINAL REPORT AND SCORES
# # ---------------------------------------------------------

# if st.session_state.interview_finished:
#     st.divider()
#     st.header("📊 Interview Evaluation Report")

#     report = st.session_state.report

#     if report:
#         summary = report.get("score_summary", {})
#         overall = summary.get("overall_percentage")

#         col1, col2, col3 = st.columns(3)

#         with col1:
#             st.metric(
#                 "Questions answered",
#                 report.get("questions_answered", 0),
#             )

#         with col2:
#             st.metric(
#                 "Answers evaluated",
#                 summary.get("scored_answers", 0),
#             )

#         with col3:
#             st.metric(
#                 "Overall score",
#                 f"{overall}%" if overall is not None else "Unavailable",
#             )

#         st.subheader("Individual Answer Scores")

#         for number, score in enumerate(
#             report.get("answer_scores", []), start=1
#         ):
#             with st.expander(
#                 f"Question {number}: "
#                 f"{score.get('total', 'N/A')} / "
#                 f"{score.get('max_total', 20)}"
#             ):
#                 st.write("**Question:**", score.get("question", ""))
#                 st.write("**Answer:**", score.get("answer", ""))
#                 display_score(score)

#         st.subheader("Downloadable Report")
#         st.json(report)

#         report_bytes = json.dumps(
#             report,
#             indent=2,
#             ensure_ascii=False,
#         ).encode("utf-8")

#         st.download_button(
#             "Download JSON Report",
#             data=report_bytes,
#             file_name="final_report.json",
#             mime="application/json",
#         )

#     else:
#         st.warning(
#             "The interview ended, but the final report could not be created. "
#             "Check the saved answers and the Groq API configuration."
#         )

#     if st.button("Start a New Interview"):
#         reset_interview()
#         st.session_state.prepared = False
#         st.session_state.questions = []
#         st.rerun()

# elif not st.session_state.prepared:
#     st.info(
#         "Upload your resume and job description in the sidebar, "
#         "then generate your interview questions."
#     )



# app.py
# Streamlit UI for the AI Mock Interview Coach

import base64
import json
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



def render_question_player_and_timer(question, question_key, audio_bytes=None, timer_seconds=60):
    """Play the question and start a browser-visible countdown only when speech ends."""
    question_json = json.dumps(str(question))
    key_json = json.dumps(str(question_key))
    if audio_bytes:
        audio_b64 = base64.b64encode(audio_bytes).decode("ascii")
        audio_markup = f'<audio id="question-audio" controls autoplay src="data:audio/wav;base64,{audio_b64}"></audio>'
        play_js = """
          const audio = document.getElementById('question-audio');
          audio.addEventListener('ended', startCountdown, {once: true});
          audio.addEventListener('error', () => {
            status.textContent = 'Audio could not play. Click Play on the audio player.';
          });
          audio.play().catch(() => {
            status.textContent = 'Browser blocked autoplay. Click Play on the audio player.';
          });
        """
    else:
        audio_markup = '<button id="read-question" style="padding:8px 12px;cursor:pointer;">🔊 Read Question Aloud</button>'
        play_js = """
          function readQuestion() {
            if (!('speechSynthesis' in window)) {
              status.textContent = 'Speech is unsupported. Click Start Answer Timer when ready.';
              startButton.style.display = 'inline-block';
              return;
            }
            window.speechSynthesis.cancel();
            const utterance = new SpeechSynthesisUtterance(questionText);
            utterance.lang = 'en-US';
            utterance.rate = 0.9;
            utterance.onstart = () => status.textContent = 'AI is reading the question…';
            utterance.onend = () => {
              status.textContent = 'Question finished. Your answer timer is starting.';
              startCountdown();
            };
            utterance.onerror = () => {
              status.textContent = 'Speech failed. Click Start Answer Timer when ready.';
              startButton.style.display = 'inline-block';
            };
            window.speechSynthesis.speak(utterance);
          }
          document.getElementById('read-question').addEventListener('click', readQuestion);
          setTimeout(readQuestion, 250);
        """
    html_doc = f"""<!doctype html>
<html><head><meta charset="utf-8"><style>
body {{font-family:Arial,sans-serif;margin:0;padding:8px;color:#222}}
.timer {{font-size:28px;font-weight:700;margin:10px 0}}
.status {{font-size:14px;margin:6px 0}}
button {{margin:4px 4px 4px 0}}
</style></head><body>
<div><strong>AI Interviewer</strong></div>
<div>{audio_markup}</div>
<div id="status" class="status">Preparing question…</div>
<div>Time remaining</div><div id="timer" class="timer">Waiting for question to finish…</div>
<button id="start-timer" style="display:none">Start Answer Timer</button>
<script>
const questionText = {question_json};
const questionKey = {key_json};
const limit = {int(timer_seconds)};
const status = document.getElementById('status');
const timer = document.getElementById('timer');
const startButton = document.getElementById('start-timer');
let started = false;
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
startButton.addEventListener('click', startCountdown);
{play_js}
</script></body></html>"""
    components.html(html_doc, height=190, scrolling=False)


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

    manager = core.InterviewStateManager()
    st.session_state.session_id = manager.create_session(
        st.session_state.role,
        st.session_state.questions,
    )


def finish_interview(status="completed"):
    st.session_state.interview_finished = True
    st.session_state.interview_started = False

    session_id = st.session_state.session_id
    if session_id:
        manager = core.InterviewStateManager()
        manager.end_interview(session_id, status=status)

    records = st.session_state.answers
    core.save_member2_answers(records, st.session_state.role)

    # Create final report from saved answers and show it in this app.
    report = core.member3_process_saved_interview()
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

    with st.expander("View generated questions"):
        for i, question in enumerate(
            st.session_state.questions, start=1
        ):
            st.write(f"{i}. {question}")

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

    question = questions[index]
    question_key = f"{st.session_state.session_id}_{index}"

    st.divider()
    st.subheader(
        f"Question {index + 1} of {len(questions)}"
    )
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

    render_question_player_and_timer(
        question=question,
        question_key=question_key,
        audio_bytes=st.session_state.tts_audio_bytes,
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
        # The recording has been received. Always attempt processing, then advance
        # the question even if STT, scoring, or a persistence service fails.
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
            with st.spinner("Processing answer: saving audio, transcribing, and recording results..."):
                core.STORAGE_DIR.mkdir(parents=True, exist_ok=True)
                recording_path = core.STORAGE_DIR / f"answer_{st.session_state.session_id}_{index + 1}.wav"
                try:
                    recording_path.write_bytes(audio_bytes)
                except Exception as exc:
                    stt_error = f"Audio file could not be saved: {exc}"

                # STT
                try:
                    transcript = (core.transcribe_audio(audio_bytes, filename=recording_path.name) or "").strip()
                    if not transcript:
                        stt_error = "STT returned an empty transcript. The recording was retained if saving succeeded."
                except Exception as exc:
                    stt_error = f"STT failed: {exc}"

                # Score only when a transcript exists.
                if transcript:
                    try:
                        score = core.score_answer(question, transcript)
                    except Exception as exc:
                        score["answer"] = transcript
                        score["evaluation_error"] = str(exc)

                record = {
                    "question_number": index + 1,
                    "stage": "Interview",
                    "category": "resume_or_job_description",
                    "question": question,
                    "answer": transcript,
                    "audio_file": str(recording_path) if recording_path else None,
                    "stt_error": stt_error,
                    "follow_up_questions": [],
                    "follow_up_answers": [],
                    "score": score,
                    "started_at": None,
                    "ended_at": core.utc_now(),
                }
                st.session_state.answers.append(record)

                try:
                    manager = core.InterviewStateManager()
                    manager.save_answer(
                        st.session_state.session_id,
                        index + 1,
                        transcript or "[STT failed; audio saved if possible]",
                        [], [], score,
                    )
                except Exception as exc:
                    st.warning(f"SQLite save failed: {exc}")

                try:
                    core.save_member2_answers(st.session_state.answers, st.session_state.role)
                except Exception as exc:
                    st.warning(f"JSON save failed: {exc}")

            # IMPORTANT: advance state before rendering optional score/details.
            # This ensures a UI/reporting issue cannot leave the same question visible.
            st.session_state.question_index = index + 1
            st.session_state.deadline = None
            st.session_state.question_speech_seconds = 0.0
            st.session_state.recording_key += 1
            st.session_state.last_spoken_question = None
            st.session_state.tts_audio_key = None
            st.session_state.tts_audio_bytes = None
            st.session_state.tts_error = None

            if st.session_state.question_index >= len(questions):
                try:
                    finish_interview("completed")
                except Exception as exc:
                    st.error(f"Answers were processed, but final report generation failed: {exc}")
                    st.session_state.interview_finished = True
                    st.session_state.interview_started = False

            if transcript:
                st.success("Answer processed. Moving to the next question.")
                st.write("**Recognized answer:**", transcript)
                try:
                    display_score(score)
                except Exception as exc:
                    st.warning(f"Score display failed, but the answer was processed: {exc}")
            else:
                st.warning("The recording was submitted, but STT did not return text. The question has still been advanced.")
                if stt_error:
                    st.caption(stt_error)
            st.rerun()

    if skip_question:
        # Save a zero-score record for the skipped question.
        empty_score = core.score_answer(question, "")
        record = {
            "question_number": index + 1,
            "stage": "Interview",
            "category": "resume_or_job_description",
            "question": question,
            "answer": "",
            "follow_up_questions": [],
            "follow_up_answers": [],
            "score": empty_score,
            "started_at": None,
            "ended_at": core.utc_now(),
        }

        manager = core.InterviewStateManager()
        manager.save_answer(
            st.session_state.session_id,
            index + 1,
            "",
            [],
            [],
            empty_score,
        )

        st.session_state.answers.append(record)
        core.save_member2_answers(
            st.session_state.answers,
            st.session_state.role,
        )

        st.session_state.question_index += 1
        st.session_state.deadline = None
        st.session_state.question_speech_seconds = 0.0
        st.session_state.recording_key += 1
        st.session_state.last_spoken_question = None

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
