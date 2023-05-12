clear
class_1 = {'Acoustic environment', 'Alarm', 'Bell', 'Deformable shell', 'Digestive', ...
 'Domestic animals, pets', 'Domestic sounds, home sounds', 'Engine', ...
 'Explosion', 'Fire', 'Generic impact sounds', 'Glass', 'Hands', ...
 'Heart sounds, heartbeat', 'Human group actions', 'Human locomotion', ...
 'Human voice', 'Liquid', 'Livestock, farm animals, working animals', ...
 'Mechanisms', 'Miscellaneous sources', 'Music genre', 'Music mood', ... 
 'Music role', 'Musical concepts', 'Musical instrument', 'Noise', ...
 'Onomatopoeia', 'Other sourceless',  ... 
 'Respiratory sounds', 'Silence', 'Sound reproduction', ...
 'Specific impact sounds', 'Surface contact', 'Thunderstorm', 'Tools', ...
 'Vehicle', 'Water', 'Whistling', 'Wild animals', 'Wind', 'Wood'};
class_2 = {'Animal', 'Channel, environment and background', 'Human sounds', 'Music', ...
 'Natural sounds', 'Sounds of things', 'Source-ambiguous sounds'};
Y1 = [1583.  944.  413.  179.  406.  655. 1845.  560.  545.  122.  459.  171. ...
  170.  110.  384.  180. 7210.  722.  774.  674.  477. 2391.  411.  508. ... 
   77. 3347.  644. 1163.  270.  555.   60.  180.   61.  325.   73. ...
  393. 1960.  708.   61. 1197.  230.  280.];
Y2 = [2469. 2313. 8026. 8738.  988. 7910. 2423.];

% Load test metrics
load('./results/eval_metrics.mat')

close all
figure(1)
X1 = categorical(class_1);

yyaxis left
plot(X1, test_mAP_1, '-kx', 'linewidth', 2, 'markersize', 10); hold off
ylim([0 1])
ylabel('Average Precision')
set(gca, 'ycolor', 'k', 'FontSize', 10)

yyaxis right
b1 = bar(X1, Y1, 'facecolor', '#0072BD', 'edgecolor', 'none'); hold on
b1.FaceAlpha=0.25;
set(gca,'YScale','log')
ylim([20, 10^4])
ylabel('Number of audio clips')
set(gca, 'ycolor', '#0072BD')
grid on, grid minor


figure(2)
X2 = categorical(class_2);

yyaxis left
plot(X2, test_mAP_2, '-kx', 'linewidth', 2, 'markersize', 10); hold off
ylabel('Average Precision')
ylim([0 1])
set(gca, 'ycolor', 'k', 'FontSize', 10)

yyaxis right
b1 = bar(X2, Y2, 'facecolor', '#0072BD', 'edgecolor', 'none'); hold on
b1.FaceAlpha=0.25;
set(gca,'YScale','log')
ylim([20, 10^4])
ylabel('Number of audio clips')
set(gca, 'ycolor', '#0072BD')
grid on, grid minor

figure(3)
X1 = categorical(class_1);

yyaxis left
plot(X1, test_AUC_1, '-kx', 'linewidth', 2, 'markersize', 10); hold off
ylabel('AUC')
ylim([0 1])
set(gca, 'ycolor', 'k', 'FontSize', 10)

yyaxis right
b1 = bar(X1, Y1, 'facecolor', '#0072BD', 'edgecolor', 'none'); hold on
b1.FaceAlpha=0.25;
set(gca,'YScale','log')
ylim([20, 10^4])
ylabel('Number of audio clips')
set(gca, 'ycolor', '#0072BD')
grid on, grid minor


figure(4)
X2 = categorical(class_2);

yyaxis left
plot(X2, test_AUC_2, '-kx', 'linewidth', 2, 'markersize', 10); hold off
ylabel('AUC')
ylim([0 1])
set(gca, 'ycolor', 'k', 'FontSize', 10)

yyaxis right
b1 = bar(X2, Y2, 'facecolor', '#0072BD', 'edgecolor', 'none'); hold on
b1.FaceAlpha=0.25;
set(gca,'YScale','log')
ylim([20, 10^4])
ylabel('Number of audio clips')
set(gca, 'ycolor', '#0072BD')
grid on, grid minor


