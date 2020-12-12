clc
clear all
close all
warning off
load Classifier;
%applying the classifer on the testing data
x=readtable('test.csv');
cnames=x.Properties.VariableNames;
image_pixels=table2array(x);
Label=[];
for mjks=1:28000
img=imbinarize(uint8(reshape(image_pixels(mjks,:),[28,28])'));
[Features]=extractHOGFeatures(img,'CellSize',[8 8]);
PredictedClass=(predict(Classifier,Features));
Label=[Label;PredictedClass];
end
feat=[Features];
k=(1:28000)';
output=[k Label];
%we will be having our table as the instance of the image and its
%respective label
op=array2table(output,'VariableNames',{'ImageId','Label'});
%we are creating an output upload.csv file
writetable(op,'Upload.csv','Delimiter',',');

function [conf_matrix] = ConfusionMatrix(pred_labels, test_target, no_of_classes)
    conf_matrix = zeros(no_of_classes, no_of_classes);
    for i=1:no_of_classes
        for j=1:no_of_classes
            conf_matrix(i,j) = length(test_target(test_target==i & pred_labels==j));
        end
    end
end
