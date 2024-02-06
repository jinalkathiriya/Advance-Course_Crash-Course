## Lab 3 - Object Identification Using Transfer learning resnet50

from imageai.Classification.Custom import ClassificationModelTrainer

model_trainer = ClassificationModelTrainer()
#The ClassificationModelTrainer class allows you to train any of the 4 
#supported deep learning algorithms (MobileNetV2 , ResNet50 , InceptionV3 
#and DenseNet121) on your own image dataset to generate your own custom models.#

model_trainer.setModelTypeAsResNet50()

#This function sets the model type of the training instance you created to 
#the ResNet50 model, which means the ResNet50 algorithm will be trained on 
#your dataset.

model_trainer.setDataDirectory(r"/content/drive/MyDrive/PetImages")
#This function accepts a string which must be the path to the folder that 
#contains the test and train subfolder of your image dataset.

model_trainer.trainModel(num_experiments=100, batch_size=32)
#This is the function that starts the training process. Once it starts,
#it will create a JSON file in the dataset/json folder which contains the 
#mapping of the classes of the dataset.

from imageai.Classification.Custom import CustomImageClassification
import os

execution_path = os.getcwd()

prediction = CustomImageClassification()
prediction.setModelTypeAsResNet50()
#This function sets the model type of the image recognition instance you 
#created to the ResNet50 model, which means you will be performing your
#image prediction tasks using the “ResNet” model model generated during your
#custom training. 

prediction.setModelPath(os.path.join(execution_path, "/content/drive/MyDrive/PetImages/models/resnet50-PetImages-test_acc_0.55869_epoch-4.pt"))
#This function accepts a string which must be the path to the model file 
#generated during your custom training and must corresponds to the model
#type you set for your image prediction instance. 

prediction.setJsonPath(os.path.join(execution_path, "/content/drive/MyDrive/PetImages/models/PetImages_model_classes.json"))
#This function accepts a string which must be the path to the JSON file
#generated during your custom training.

prediction.loadModel()
#This function loads the model from the path you specified in the function
#call above into your image prediction instance. You will have to set the
#parameter num_objects to the number of classes in your image dataset.

predictions, probabilities = prediction.classifyImage("/content/drive/MyDrive/PetImages/test/dog/dog-test-images/9364.jpg", result_count=2)
#This is the function that performs actual prediction of an image.
#It can be called many times on many images once the model as been 
#loaded into your prediction instance.

print(predictions, probabilities)