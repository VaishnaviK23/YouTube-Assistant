import textwrap
import streamlit as st
from youtube_assistant import vector_db_from_video, get_response_from_query

st.title("YouTube Assistant")

video_url = st.text_input("Enter YouTube URL")
query = st.text_input("Enter your question about the video")
submit = st.button("Analyze Video")

if video_url and query and submit:
    vector_store = vector_db_from_video(video_url)
    response = get_response_from_query(vector_store, query, 4)
    st.subheader("Answer:")
    st.text(textwrap.fill(response))
