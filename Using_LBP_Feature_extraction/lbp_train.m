clc
clear all
close all
warning off

%training our model
x=readtable('lbptrain.csv');
labels=x(:,1);
trainingFeatures=[];% to store the features we get after hog feature extrtaction
trainingLabels=labels;       
image_pixels=(x(:,2:end));%image pixel values are  from 2nd row to the end of the column
cnames=string(image_pixels.Properties.VariableNames);
image_pixels=table2array(x(:,2:end));
for i=1:20000
%feature scaling by converting grayscale to binary using global threshold    
ms=imbinarize(uint8(reshape(image_pixels(i,:),[28,28])'));
%using LBP(Linear binary pattern) feature extraction
trainingFeatures(i,:)=extractLBPFeatures(ms,'CellSize',[8 8],'Upright',false,'Radius',3,'Normalization','None','Interpolation','Nearest');
% M=feature_vec(ms);
end
Classifier=fitcecoc(trainingFeatures,table2array(trainingLabels));
save Classifier Classifier;
