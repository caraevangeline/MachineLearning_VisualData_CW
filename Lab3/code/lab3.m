%%Information
% facial age estimation
% regression method: linear regression

%% settings
clear;
clc;

% path 
database_path = './data_age.mat';
result_path = './results/';

% initial states
absTestErr = 0;
cs_number = 0;


% cumulative error level
error_level = 5;

%% Training 
load(database_path);

nTrain = length(trData.label); % number of training samples
nTest  = length(teData.label); % number of testing samples
xtrain = trData.feat; % feature
ytrain = trData.label; % labels

w_lr = regress(ytrain,xtrain);
   
%% Testing
xtest = teData.feat; % feature
ytest = teData.label; % labels

y_LR = xtest * w_lr;

%% Compute the MAE and CS value (with cumulative error level of 5) for linear regression 
%------Mean Absolute Error-------
mae_LR = 0;
for i = 1:502
    mae_LR = mae_LR + abs(ytest(i) - y_LR(i));
end
mae_LR = mae_LR/502;
%-----Cumulative Error----------
c = 0;
for i = 1:502
    error = abs(ytest(i) - y_LR(i));
    if error <= error_level
        c = c+1;
    end
end
cs_LR_5 = (c/502)*100;
%% Generate a cumulative score (CS) vs. error level plot by varying the error level from 1 to 15. The plot should look at the one in the Week6 lecture slides
% for error_level = 1:15
%     c = 0;
%     for i = 1:502
%        error = abs(ytest(i) - y_LR(i));
%        if error <= error_level
%           c = c+1;
%        end
%     end
%     cs_LR(error_level) = (c/502);
% end
% plot(1:15,cs_LR);
% xlabel('Level');
% ylabel('CS');
% title('Error Level plot');
%% Compute the MAE and CS value (with cumulative error level of 5) for both partial least square regression and the regression tree model by using the Matlab built in functions.
%%%%%----- Partial Least Square Regression -----%%%%%%%%%%
[XL,yl,XS,YS,beta,PCTVAR] = plsregress(xtrain,ytrain,10);
y_PLS = [ones(size(xtest,1),1) xtest]*beta;
%------Minimum Absolute Error-----------------
mae_PLS = 0;
for i = 1:502
    mae_PLS = mae_PLS + abs(ytest(i) - y_PLS(i));
end
mae_PLS = mae_PLS/502;
%-----Cumulative Error----------
c = 0;
for i = 1:502
    error = abs(ytest(i) - y_PLS(i));
    if error <= error_level
        c = c+1;
    end
end
cs_PLS_5 = (c/502)*100;
%%%%%----- Regression Tree ---------%%%%%%%%%%
Mdl = fitrtree(xtrain, ytrain);
y_RT = predict(Mdl, xtest);
%------Minimum Absolute Error-----------------
mae_RT = 0;
for i = 1:502
    mae_RT = mae_RT + abs(ytest(i) - y_RT(i));
end
mae_RT = mae_RT/502;
%-----Cumulative Error----------
c = 0;
for i = 1:502
    error = abs(ytest(i) - y_RT(i));
    if error <= error_level
        c = c+1;
    end
end
cs_RT_5 = (c/502)*100;
%% Compute the MAE and CS value (with cumulative error level of 5) for Support Vector Regression by using LIBSVM toolbox
run('libsvm-3.14/matlab/make')
bestc=1024;bestg=2.8248; %Calculated best values after running the below part of the program
bestcv=0;
% tic 
% for log2c = -1:10
%   for log2g = -1:0.1:1.5
%     cmd = ['-v 5 -t 1 -c ', num2str(2^log2c), ' -g ', num2str(2^log2g)];
%     cv = svmtrain(ytrain, xtrain, cmd);
%     if (cv >= bestcv)
%       bestcv = cv; bestc = 2^log2c; bestg = 2^log2g;
%     end
%     fprintf('%g %g %g (best c=%g, g=%g, rate=%g)\n', log2c, log2g, cv, bestc, bestg, bestcv);
%   end
% end
% toc
options=sprintf('-s 4 -t 1 -c %f -b 1 -g %f -q', bestc, bestg);
model=svmtrain(ytrain, xtrain,options);
[y_SVM, accuracy , dec_values] = svmpredict(ytest,xtest, model,'-b 1');
%------Minimum Absolute Error-----------------
mae_SVM = 0;
for i = 1:502
    mae_SVM = mae_SVM + abs(ytest(i) - y_SVM(i));
end
mae_SVM = mae_SVM/502;
%-----Cumulative Error----------
c = 0;
for i = 1:502
    error = abs(ytest(i) - y_SVM(i));
    if error <= error_level
        c = c+1;
    end
end
cs_SVM_5 = (c/502)*100;

%------------------------------------------------------------------------
%%%%%%%%%%%%%Comparison between different models%%%%%%%%%%%%%%%%%%
for error_level = 1:15
    c1 = 0;
    c2 = 0;
    c3 = 0;
    c4 = 0;
    for i = 1:502
       error1 = abs(ytest(i) - y_LR(i));
       error2 = abs(ytest(i) - y_PLS(i));
       error3 = abs(ytest(i) - y_RT(i));
       error4 = abs(ytest(i) - y_SVM(i));
       if error1 <= error_level
          c1 = c1+1;
       end
       if error2 <= error_level
          c2 = c2+1;
       end
       if error3 <= error_level
          c3 = c3+1;
       end
       if error4 <= error_level
          c4 = c4+1;
       end
    end
    cs_LR(error_level) = (c1/502);
    cs_PLS(error_level) = (c2/502);
    cs_RT(error_level) = (c3/502);
    cs_SVM(error_level) = (c4/502);
end
plot(1:15,cs_LR);
hold on
plot(1:15,cs_PLS);
plot(1:15,cs_RT);
plot(1:15,cs_SVM);
hold off
%plot(1:15,cs_LR, 1:15,cs_PLS, 1:15,cs_RT, 1:15,cs_SVM);
legend('Linear Regression', 'Partial LS Regression', 'Regression Tree', 'SVM', 'Location', 'SouthEast')
xlabel('Level');
ylabel('CS');
title('Error Level plot');