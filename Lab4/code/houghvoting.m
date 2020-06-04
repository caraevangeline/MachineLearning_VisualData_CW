function hough_array = houghvoting(patches,position,spa_scale,tem_scale,frame_num,flag_mat,struct_cb)
% hough_array           : is a matrix with 7 rows, each column indicating the
%                       : predicted(votes) spatial location, predicted (voted) start and end frames,
%                       : the votes, bounding box values(scale compensated). Example: Let a descriptor matched with
%                       : the codeword (with one offset) cast votes at the spatial location [x,y],
%                       : temporal location [s,e] (predictions for start and end frames), bounding box values [b1 b2] (stored during training)
%                       : with value(weight) 'v', then corresponding column in the
%                       : matrix hough_array will be [x y s e v b1 b2]'.
% patches               : i/p descriptors
% position              : spatial location of the detected STIP                   
% spa_scale             : spatial scale                      
% tem_scale             : temporal scale
% frame_num             : frame number at which STIP was detected
% flag_mat              : refer to ism_test_voting.m
  
%-----------------------------------------------------------------------------------------------------
% Write your code here to compute the matrix hough_array
%-----------------------------------------------------------------------------------------------------
index = 1; 
for i = 1:size(patches,2)
    for j = 1:size(flag_mat,1)
        match = sum(flag_mat(:,i));
        if flag_mat(j,i) == 1
            z = struct_cb.offset(j).tot_cnt;
            for k = 1:z
                hough_array(1,index) = position(1,i) - (struct_cb.offset(j).spa_offset(1,k) * spa_scale(i));
                hough_array(2,index) = position(2,i) - (struct_cb.offset(j).spa_offset(2,k) * spa_scale(i));
                hough_array(3,index) = frame_num(i) - (tem_scale(i) * struct_cb.offset(j).st_end_offset(1,k)) ;
                hough_array(4,index) = frame_num(i) - (tem_scale(i) * struct_cb.offset(j).st_end_offset(2,k));
                hough_array(5,index) = (1/z)*(1/match);
                hough_array(6,index) = struct_cb.offset(j).hei_wid_bb(1,k) * spa_scale(i) ;
                hough_array(7,index) = struct_cb.offset(j).hei_wid_bb(2,k) * spa_scale(i);
                index = index +1;
            end
        end
    end
end
save('hough_array','hough_array');
%save('match','match');               
% count = 0;
% for i = 1:size(flag_mat,2)
%     for j = 1:size(flag_mat,1)
%         if flag_mat(j, i) == 1
%             count = count + struct_cb.offset(j).tot_cnt;
%         end
%     end
% end
% save('count', 'count')
% hough_array = flag_mat(1:7,:);
save('flag_mat','flag_mat'); 
