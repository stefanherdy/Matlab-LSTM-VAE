load('91301486_20190826_121414__M5_Log_.mat')
fig1 = figureGen(15,20)
subplot(9,1,1)
%time = TT.Properties.RowTimes
plot(data.Time, data.Depth, 'k')
title('Time Series Data Point 91301486_20190826_121414__M5_Log, Seestadt Aspern')
ylh = ylabel('Depth [m]')
set(ylh,'rotation',0,'VerticalAlignment','middle', 'HorizontalAlignment','right')
grid on

subplot(9,1,2)
plot(data.Time, data.FeedRate, 'k')
ylh = ylabel('Feedrate [1/s]')
set(ylh,'rotation',0,'VerticalAlignment','middle', 'HorizontalAlignment','right')
grid on

subplot(9,1,3)
plot(data.Time, data.PullDownForce, 'k')
ylh = ylabel('PullDownForce [N]')
set(ylh,'rotation',0,'VerticalAlignment','middle', 'HorizontalAlignment','right')
grid on

subplot(9,1,4)
plot(data.Time, data.VibratorAmperage, 'k')
ylh = ylabel('VibratorAmperage [A]')
set(ylh,'rotation',0,'VerticalAlignment','middle', 'HorizontalAlignment','right')
grid on

subplot(9,1,5)
plot(data.Time, data.VibratorFrequency, 'k')
ylh = ylabel('VibratorFrequency [Hz]')
set(ylh,'rotation',0,'VerticalAlignment','middle', 'HorizontalAlignment','right')
grid on

subplot(9,1,6)
plot(data.Time, data.VibratorTemperature, 'k')
ylh = ylabel('VibratorTemperature [°C]')
set(ylh,'rotation',0,'VerticalAlignment','middle', 'HorizontalAlignment','right')
grid on

subplot(9,1,7)
plot(data.Time, data.InclinationX, 'k')
ylh = ylabel('InclinationX [°]')
set(ylh,'rotation',0,'VerticalAlignment','middle', 'HorizontalAlignment','right')
grid on

subplot(9,1,8)
plot(data.Time, data.InclinationY, 'k')
ylh = ylabel('InclinationY [°]')
set(ylh,'rotation',0,'VerticalAlignment','middle', 'HorizontalAlignment','right')
grid on

subplot(9,1,9)
plot(data.Time, data.WeightNet, 'k')
ylh = ylabel('Weight [10³ kg]')
set(ylh,'rotation',0,'VerticalAlignment','middle', 'HorizontalAlignment','right')
grid on
get(gca,'OuterPosition')
plotTidyUpStack()

xlabel('Time')
%plotTT(data)

