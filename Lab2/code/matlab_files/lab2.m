%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%coursework: face recognition with eigenfaces
% need to replace with your own path
addpath software;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 1 Loading of the images: You need to replace the directory 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Imagestrain = loadImagesInDirectory ( 'images/training-set/23x28/');
[Imagestest, Identity] = loadTestImagesInDirectory ( 'images/testing-set/23x28/');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 2&3 Computation of the mean, the eigenvalues, amd the eigenfaces stored in the facespace:
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ImagestrainSizes = size(Imagestrain);
Means = floor(mean(Imagestrain));
CenteredVectors = (Imagestrain - repmat(Means, ImagestrainSizes(1), 1));
CovarianceMatrix = cov(CenteredVectors);
[U, S, V] = svd(CenteredVectors,'econ');
Space = V(: , 1 : ImagestrainSizes(1))';
Eigenvalues = diag(S);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 4 Display of the mean image:
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
MeanImage = uint8 (zeros(28, 23));
for k = 0:643
   MeanImage( mod (k,28)+1, floor(k/28)+1 ) = Means (1,k+1);
end
figure;
subplot (1, 1, 1);
imshow(MeanImage);
title('Mean Image');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 5 Display of the 20 first eigenfaces : Write your code here
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
A = V';
for k = 1:20
  min1 = min(A(k,:));
  max1 = max(A(k,:));
  for j = 1:644
      newImage(k,j) = 255*(A(k,j) - min1)/(max1-min1);
  end
end
Eigenface = uint8 (zeros(28, 23));
for i = 1:20
   for k = 0:643
       Eigenface(mod (k,28)+1, floor(k/28)+1 ) = newImage(i,k+1);       
   end
   subplot (4, 5, i);
   imshow(Eigenface);
   title(['Eigenface ', num2str(i)]);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 6 Projection of the two sets of images onto the face space:
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Locationstrain=projectImages (Imagestrain, Means, Space);
Locationstest=projectImages (Imagestest, Means, Space);
Threshold =20;
TrainSizes=size(Locationstrain);
TestSizes = size(Locationstest);
Distances=zeros(TestSizes(1),TrainSizes(1));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 7 Distances contains for each test image, the distance to every train image.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i=1:TestSizes(1)
    for j=1: TrainSizes(1)
        Sum=0;
        for k=1: Threshold
            Sum=Sum+((Locationstrain(j,k)-Locationstest(i,k)).^2);
        end
        Distances(i,j)=Sum;
    end
end
Values=zeros(TestSizes(1),TrainSizes(1));
Indices=zeros(TestSizes(1),TrainSizes(1));
for i=1:70
    [Values(i,:), Indices(i,:)] = sort(Distances(i,:));
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 8 Display of first 6 recognition results, image per image:
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure;
x=6;
y=2;
c=1;
for i=1:6
      Image = uint8 (zeros(28, 23));
      for k = 0:643
          Image( mod (k,28)+1, floor(k/28)+1 ) = Imagestest (i,k+1);
      end
      subplot (x,y,2*c-1);
      imshow (Image);
      title('Image tested');
      Imagerec = uint8 (zeros(28, 23));
      for k = 0:643
          Imagerec( mod (k,28)+1, floor(k/28)+1 ) = Imagestrain ((Indices(i,1)),k+1);
      end
      subplot (x,y,2*c);
      imshow (Imagerec);
      title(['Image recognised with ', num2str(Threshold), ' eigenfaces:',num2str((Indices(i,1))) ]);
      c=c+1;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 9 recognition rate compared to the number of test images: Write your code here to compute the recognition rate using top 20 eigenfaces.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
recognised_person=zeros(1,40);
recognitionrate=zeros(1,5);
number_per_number=zeros(1,5);
number_of_test_images=zeros(1,40);% Number of test images of one given person.%YY I modified here
  for i=1:70
      number_of_test_images(1,Identity(1,i))= number_of_test_images(1,Identity(1,i))+1;%YY I modified here
      [Values(i,:), Indices(i,:)] = sort(Distances(i,:));
  end
i=1;
while (i<70)
    id=Identity(1,i);   
    distmin=Values(id,1);
    indicemin=Indices(id,1);
    while (i<70)&&(Identity(1,i)==id)
        if (Values(i,1)<distmin)
            distmin=Values(i,1);
            indicemin=Indices(i,1);
        end
        i=i+1;    
    end
    recognised_person(1,id)=indicemin;
    number_per_number(number_of_test_images(1,id))=number_per_number(number_of_test_images(1,id))+1;
    if (id==floor((indicemin-1)/5)+1) %the good personn was recognised
        recognitionrate(number_of_test_images(1,id))=recognitionrate(number_of_test_images(1,id))+1;        
    end
end
for  i=1:5
   recognitionrate(1,i)=recognitionrate(1,i)/number_per_number(1,i);
end
RR = mean(recognitionrate(1,:));
figure;
plot (recognitionrate(1,:));
title('Recognitionrate for 20');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
% 10 effect of threshold (i.e. number of eigenfaces):   
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
averageRR=zeros(1,20);
for t=1:40
  Threshold =t;  
  Distances=zeros(TestSizes(1),TrainSizes(1));
  for i=1:TestSizes(1)
    for j=1: TrainSizes(1)
        Sum=0;
        for k=1: Threshold
            Sum=Sum+((Locationstrain(j,k)-Locationstest(i,k)).^2);
        end
        Distances(i,j)=Sum;
    end
  end
  Values=zeros(TestSizes(1),TrainSizes(1));
  Indices=zeros(TestSizes(1),TrainSizes(1));
  number_of_test_images=zeros(1,40);% Number of test images of one given person.%YY I modified here
  for i=1:70
      number_of_test_images(1,Identity(1,i))= number_of_test_images(1,Identity(1,i))+1;%YY I modified here
      [Values(i,:), Indices(i,:)] = sort(Distances(i,:));
  end
  recognised_person=zeros(1,40);
  recognitionrate=zeros(1,5);
  number_per_number=zeros(1,5);
  i=1;
  while (i<70)
    id=Identity(1,i);   
    distmin=Values(id,1);
    indicemin=Indices(id,1);
    while (i<70)&&(Identity(1,i)==id) 
        if (Values(i,1)<distmin)
            distmin=Values(i,1);
            indicemin=Indices(i,1);
        end
        i=i+1;    
    end
    recognised_person(1,id)=indicemin;
    number_per_number(number_of_test_images(1,id))=number_per_number(number_of_test_images(1,id))+1;
    if (id==floor((indicemin-1)/5)+1) %the good personn was recognised
        recognitionrate(number_of_test_images(1,id))=recognitionrate(number_of_test_images(1,id))+1;        
    end
  end
  for  i=1:5
    recognitionrate(1,i)=recognitionrate(1,i)/number_per_number(1,i);
  end
  averageRR(1,t)=mean(recognitionrate(1,:));
end
figure;
plot(averageRR(1,:));
title('Recognition rate against the number of eigenfaces used');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 11 effect of K: You need to evaluate the effect of K in KNN and plot the recognition rate against K. Use 20 eigenfaces here.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
averageRR=zeros(1,20);
Threshold = 20;  %No of eigenfaces
Distances=zeros(TestSizes(1),TrainSizes(1));
%--------Calculates the distance of all test images to train images------
for i=1:TestSizes(1)
    for j=1: TrainSizes(1)
        Sum=0;
        for k=1: Threshold
            Sum=Sum+((Locationstrain(j,k)-Locationstest(i,k)).^2);
        end
        Distances(i,j)=Sum;
    end
end
%--------Sorts the Distances and the values of distances are in Values
%and the corresponding train image index is in Indices
Values=zeros(TestSizes(1),TrainSizes(1));
Indices=zeros(TestSizes(1),TrainSizes(1));
for i=1:70 
    [Values(i,:), Indices(i,:)] = sort(Distances(i,:));
end
%-------Nearby values are concatenated to limited values by putting them in
%40 clusters (ie. every person has 40 sets of images)
person=zeros(70,200);
person(:,:)=floor((Indices(:,:)-1)/5)+1;
%------Loop Run for KNN from k=1 to k=20--------------- 
for K=1:20
   recog_person = zeros(1,70);
   recognitionrate = 0;
   number_of_occurance=zeros(70,K);
   %-----Loop run for every test image----------------
   for i=1:70
      max=0;
      %---------Loop run to check the k nearest occurances------
      for j=1:K
         for k=j:K
            if (person(i,k)==person(i,j))
                number_of_occurance(i,j)=number_of_occurance(i,j)+1;
            end
         end
         if (number_of_occurance(i,j)>max)
            max=number_of_occurance(i,j);
            jmax=j;
         end
      end
     recog_person(1,i)=person(i,jmax);
     %--------------If the identity matches increment the recongnitionrate--
     if (Identity(1,i)==recog_person(1,i))
         recognitionrate=recognitionrate+1;
     end
     averageRR(1,K)=recognitionrate/70;
   end
end
figure;
plot(averageRR(1,:));
title('Recognition rate for different values of k in KNN');

