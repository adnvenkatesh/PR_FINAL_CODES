save Classifier Classifier;

x=readtable('test.csv');

cnames=x.Properties.VariableNames;
orig_table=table2array(x); 
%copied the testing labels into testLabels for later comparison
testLabels=orig_table(:,785);
%deleting the last column

x(:,end)=[ ] ;

image_pixels=table2array(x);
Label=[];
for mjks=1:1000
img=imbinarize(uint8(reshape(image_pixels(mjks,:),[28,28])'));
[Features]=extractHOGFeatures(img,'CellSize',[8 8]);
PredictedClass=(predict(Classifier,Features));
Label=[Label;PredictedClass];
end

conf_matrix=ConfusionMatrix(Label,testLabels,10);
disp(conf_matrix)

a=trace(conf_matrix);
b=sum(conf_matrix,'all');
accuracy=a/b*100;
disp('Accuracy:')
disp(accuracy)

function conf_matrix=ConfusionMatrix(pred_labels, test_target, no_of_classes)
    conf_matrix = zeros(no_of_classes, no_of_classes);
    for i=1:no_of_classes
        for j=1:no_of_classes
            conf_matrix(i,j) = length(test_target(test_target==i & pred_labels==j));
        end
    end
end


