FROM python:3.7
EXPOSE 8501
COPY . /
RUN pip3 install -r requirements.txt
CMD streamlit run streamlit_index.py