#  Age-Gender Detector

- [Age-Gender Detector](#age-gender-detector)
  - [Data](#data)
  - [Model Cropped](#model-cropped)
- [Dash images](#dash-images)
    - [Información de los datos](#información-de-los-datos)
  

## Data

[UTKFace dataset](https://susanqq.github.io/UTKFace/)

[UTKFace dataset Kaggle](https://www.kaggle.com/jangedoo/utkface-new)

This dataset contains over 20,000 face images with annotations of
age, gender, and ethnicity. The images cover large variation in pose, facial expression, 
illumination, occlusion, resolution, etc.




## Model Cropped

We have used Convolutional Neural Networks (CNN) to predict the age and gender of the input image.

We have obtained the following metrics:
- Validation:
  - Gender accuracy: 0.905
  - Age MSE: 78.956
- Test:
  - Gender accuracy: 0.915
  - Age MSE: 75.28

# Dash images

Datasets data:
![datasets](example_imgs/datasets.png)

Model metrics (best model selected):
![metrics](example_imgs/metrics.png)

Prediction tool (the age has an error margin):
![prediction](example_imgs/prediction.png)

Saliency maps:
![saliency_maps](example_imgs/saliency_maps.png)


### Información de los datos

You need to know that if a photo is called __34_0_0_201701171712010149082.jpg.chip.jpg__, it means that the age of the individual is 34 and his gender is male. 
That is, the photo names follow the following scheme __age_gender_race_relevant_data.jpg.chip.jpg__.

Gender being 0 for male and 1 for female.
