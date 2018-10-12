clc;clear all; close all;
% data
ratio = [15:5:30];
cnn1 =  [89.4	96.8	94.7	96.5];
svm1 = [55.2	67.9	56.7	58.4];
nn1 =[85.9	85.2	91.5	89.4];
Threenn1 = [69.0 69.4 87.3 90.8 ];
fournn1 =[48.6 72.5 82.0 80.3];

cnn2 =  [91.9	94	95.4	96.1];
svm2 = [53.5	66.9	69	67.2];
nn2 =[84.1	84.2	90.8	90.9];
Threenn2 = [75.3 79.2 89.8 92.6];
fournn2 =[55.3 73.9 85.2 82.8];

cnn3 =  [84.1	91.9	94	97.2];
svm3 = [54.1	67.1	76.8	80.2];
nn3 =[77.4	81.6	89	90.4];
Threenn3 = [74.2 80.9 83.0 85.5];
fournn3 =[50.5 75.6 77.7 77.0 ];


cnn4 =  [86.3	91.5	94	95.1];
svm4 = [43.3	60.2	62.7	64.4];
nn4 =[78.2	81.3	86.7	88];
Threenn4 = [73.2 78.2 84.9 88.4];
fournn4 =[47.5 70.4 82.0 83.5];

cnn0 =  [87.925	93.55	94.525	96.225];
svm0 = [51.525	65.525	66.3	67.55];
nn0 =[81.4	83.075	89.5	89.675];

cnn = [cnn1;cnn2;cnn3;cnn4];
svm = [svm1;svm2;svm3;svm4];
nn = [nn1;nn2; nn3; nn4];
Threenn = [Threenn1;Threenn2; Threenn3; Threenn4];
fournn = [fournn1;fournn2; fournn3; fournn4];
% plot the figures
num_classifier = 3;
num_faultType = 4;
font_size =12;
marker_type ={'bo-', 'r*-.', 'ks--','b*--','rs-','kd-.','mo--'};
fault_type ={'TP', 'LG', 'DLG', 'LL'};
legend_text = {'2-NN','3-NN','4-NN'};
position_array=[0.1 0.64 0.38 0.29;
                0.57 0.64 0.38 0.29;
                0.1 0.18 0.38 0.29;
                0.57 0.18 0.38 0.29;
                0.25 0.02 0.50 0.04];
figure('units','pixels','position',[50 300 600 500]);
set(gca,'fontName','Times New Roman') 
for j=1:num_faultType
    h=subplot(2,2,j);
    set(h,'Units','normalized','Position',position_array(j,:));
    plot(ratio,nn(j,:),  marker_type{1}, 'linewidth', 1.5);
    hold on; 
    plot(ratio,Threenn(j,:),  marker_type{2}, 'linewidth', 1.5);
    hold on;  
    plot(ratio,fournn(j,:),  marker_type{3}, 'linewidth', 1.5);
    hold on; 
    xlabel('Percentage of Measured Buses (%)','fontname','Times New Roman','fontsize',font_size); 
    ylabel('LAR (%)','fontname','Times New Roman','fontsize',font_size); 
    xlim([ratio(1) ratio(end)]);
    set(gca,'XTick',ratio);
    title(fault_type{j},'fontsize',font_size);
end
h = legend(legend_text, 'Orientation',...
    'horizonal','Location','none','fontname','Times New Roman','fontsize',font_size);
legend('boxoff');
set(h, 'Units','normalized','Position', position_array(5,:));

 



% figure;
% plot(ratio,cnn1,'r-*' ,'lineWidth', 2);
% hold on;
% plot(ratio,svm1, 'b-<', 'lineWidth', 2);
% hold on;
% plot(ratio,nn1, 'g-o', 'lineWidth', 2);
% xlabel('The Ratio of Measured Buses ( %)')
% ylabel('LAR of Different Classifiers (%)')
 