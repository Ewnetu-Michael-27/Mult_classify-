#About Me
import streamlit as st
from PIL import Image


st.title("Project Lead")

col_1, col_2=st.columns(2)

with col_1:
    image=Image.open("M1.png")
    st.image(image)
with col_2:
    image_1=Image.open("M2.png")
    st.image(image_1)


st.markdown('''Greetings, My name is Michael Ewnetu and I am currently studying Data Science at MSU. 
            I did my undergrad at University of Dallas with a major in Biochemistry and minor in Computer Science.
            After that, for couple of years, I worked as a Research Scientist in a Neuroscience lab. I applied data science in behavioural research, which led to 
            a major publication on a novel protein involved in stress response. I am motivated by the great possibilites of AI and how it can be leveraged to help 
            social and ecomic facotrs of developing nations. 
        ''')

st.write("Let's connect in [Linkedin](https://www.linkedin.com/in/michaelewnetu27/)")