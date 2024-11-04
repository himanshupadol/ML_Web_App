import numpy
import pickle
import streamlit as st
from PIL import Image
import warnings


# loading the saved model
# model_01 = pickle.load(open('knnmodel_file2.sav', 'rb'))

with open('knnmodel_file.pkl', 'rb') as file:
    model_01 = pickle.load(file)

# creating a function
def digit_class(input_pic):
    
    input_pic = input_pic.convert('L') #converting the image to grayscale

    input_pic = input_pic.resize((8, 8)) #resizing the image

    pic2array = numpy.array(input_pic) #converting image to a numpy array

    pic2array_norm = 16 - (pic2array // 16) #the images in trained dataset are on color scale of 0-16, hence converting the input image similarly

    pic2array_norm = pic2array_norm.flatten().reshape(1, -1)

    prediction = model_01.predict(pic2array_norm)
    
    return f'The Digit in the input image is: {prediction[0]}'
  
def main():
    
    # giving a title
    st.title('Identify the DIGIT from its Image!')
    
    # getting the input data from the user
    file_01 = st.file_uploader("Upload Image of the Digit", type=["jpg", "jpeg", "png"])
    if file_01 is not None: # Open the uploaded image file
        image = Image.open(file_01)

        # Display the image
        st.image(image, caption='Uploaded Image', use_column_width=True)

        identify_01 = ''

        if st.button('Identify My Image'): # creating a button for Prediction
            identify_01 = digit_class(image)
            if identify_01:  # Ensure that the prediction is not None
                st.success(identify_01)
            else:
                st.error('An error occurred during the identification.')
            #st.success(identify_01)
    
    
if __name__ == '__main__':
    main()
    
