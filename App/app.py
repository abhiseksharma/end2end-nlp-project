# Core pkgs
import streamlit as st
import altair as alt

# EDA pkgs
import pandas as pd
import numpy as np

#Utils
import joblib

# Functions
pipe_lr = joblib.load(open('models/emotion_classifier_pipe_lr_07_oct_2022.pkl', 'rb'))

#Function to predict emotion
def predict_emotions(docx):
	result = pipe_lr.predict([docx])
	return result[0]

def predict_proba(docx):
	result = pipe_lr.predict_proba([docx])
	return result


emotions_emoji_dict = {"anger":"😠","disgust":"🤮", "fear":"😨😱", "happy":"🤗", "joy":"😂", "neutral":"😐", "sad":"😔", "sadness":"😔", "shame":"😳", "surprise":"😮"}


def main():
	st.title("Emotion Classifier")
	menu = ["Home", "Monitor", "About"]
	choice = st.sidebar.selectbox("Menu", menu)

	if choice == "Home":
		st.subheader("Home-Emotion In Text")

		with st.form(key='emotion_clf_form'):
			raw_text = st.text_area("type here")
			submit_text = st.form_submit_button(label='Submit')

		if submit_text:
			col1,col2 = st.columns(2)

			# Aplly functions here
			prediction = predict_emotions(raw_text)
			probability = predict_proba(raw_text)

			with col1:
				st.success("Original Text")
				st.write(raw_text)

				st.success("Prediction")
				emoji = emotions_emoji_dict[prediction]
				st.write("{}:{}".format(prediction, emoji))
				st.write("Confidence:{}".format(np.max(probability)))

			with col2:
				st.success("Prediction probability")
				# st.write(probability)
				classes = pipe_lr.classes_
				pro_df = pd.DataFrame(probability, columns=classes)
				# st.write(pro_df.T)
				pro_df_clean = pro_df.T.reset_index()
				pro_df_clean.columns = ["emotions", "probability"]

				fig = alt.Chart(pro_df_clean).mark_bar().encode(x='emotions', y='probability',color='emotions')
				st.altair_chart(fig,use_container_width=True)




	elif choice == "Monitor":
		st.subheader("Monitor App")

	else:
		st.subheader("About")




if __name__ == '__main__':
	main()