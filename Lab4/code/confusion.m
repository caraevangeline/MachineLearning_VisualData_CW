function mis_class_error = confusion()
load('struct_TP_FP');
incorrect = 0;
for i =1:3
    for j = 1:10
        flag =0;
        a = (struct_TP_FP.class(i).seq(j).array(1,1)~=1);
        b = (struct_TP_FP.class(i).seq(j).array(3,1)~=i);
        if (a)
            for k = 2:size(struct_TP_FP.class(i).seq(j).array,2)
                c = (struct_TP_FP.class(i).seq(j).array(1,k)==1);
                d = (struct_TP_FP.class(i).seq(j).array(3,k));
                if (c && d ~= i)
                   flag = 1;
                   break;
                end
                if (c && d == i)
                   flag = 2;
                   break;
                end
            end
        end
        if flag == 2
            incorrect = incorrect - 1;
        end     
        if( a || b)
            incorrect = incorrect + 1;
        end
       
    end
end
confusion = zeros(3,3);
for i = 1:3
    for j = 1:10
        for k = 1:size(struct_TP_FP.class(i).seq(j).array,2)
            a = (struct_TP_FP.class(i).seq(j).array(1,k));
            b = (struct_TP_FP.class(i).seq(j).array(3,k));
            if( a == 1)
               confusion(i,b) = confusion(i,b) + 1;
               break;
            end
        end
    end
end
save('confusion','confusion');
mis_class_error = incorrect/30;
fprintf('Misclassification rate  = %f\n', mis_class_error);
