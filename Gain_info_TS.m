x = linspace(1,100,100);
y = 0.5.*x.*sin(3.*x)+cos(7/5.*x)
%y = 0.03.*x + (randn(1,100)+3)
%z = 0.03.*x +3

ts1 = timeseries(y,1:100);

ts1.Name = 'Data with Noise';
ts1.TimeInfo.Units = 'seconds';
%ts1.TimeInfo.StartDate = '01-Jan-2011';     % Set start date.
%ts1.TimeInfo.Format = 'ss mm, hh';       % Set format for display on x-axis.

ts1.Time = ts1.Time - ts1.Time(1);    % Express time relative to the start date.

%ts2 = timeseries(z,1:100);

%ts2.Name = 'Data with Noise';
%ts2.TimeInfo.Units = 'seconds';
%ts1.TimeInfo.StartDate = '01-Jan-2011';     % Set start date.
%ts2.TimeInfo.Format = 'ss mm, hh';       % Set format for display on x-axis.

%ts2.Time = ts2.Time - ts2.Time(1);   

fig1 = figureGen()
plot(ts1,'b')
hold on
%plot(x,z,'r')
%legend('Noisy Data', 'Approximation', 'Location', 'Southeast')

