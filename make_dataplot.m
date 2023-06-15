%load('91301486_20190826_121414__M5_Log_.mat')
fig1 = figureGen(2,12)
%subplot(9,1,1)
%time = TT.Properties.RowTimes
plot(data.Time, data.Depth, 'k')
%title('Time Series Data Point 91301486_20190826_121414__M5_Log, Seestadt Aspern')
ylh = ylabel('Depth [m]')
%set(ylh,'rotation',0,'VerticalAlignment','middle', 'HorizontalAlignment','right')
grid on



xlabel('Time')
%plotTT(data)

