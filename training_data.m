clc
clear all
close all
warning off

%training our model
x=readtable('train.csv');
labels=x(:,1);
trainingFeatures=[];% to store the features we get after hog feature extrtaction
trainingLabels=labels;       
image_pixels=(x(:,2:end));%image pixel values are  from 2nd row to the end of the column
cnames=string(image_pixels.Properties.VariableNames);
image_pixels=table2array(x(:,2:end));
for i=1:42000
%feature scaling by converting grayscale to binary using global threshold    
ms=imbinarize(uint8(reshape(image_pixels(i,:),[28,28])'));
%using HOG feature extraction
trainingFeatures(i,:)=extractHOGFeatures(ms,'CellSize',[8 8]);
end
Classifier=fitcecoc(trainingFeatures,table2array(trainingLabels));
save Classifier Classifier;
