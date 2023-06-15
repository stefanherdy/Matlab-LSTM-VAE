
for i = 1:635
    maxLossList(i) = round(maxLossList(i)*100)/100;
end

diag = []

for i=1:150
    maxv = 0.5+i/100;
    right = 0;
    ges = 0;
    for j = 1:635
        if maxLossList(j) == maxv
            if maxLossList(j) > 1,23
                maxLossList(j) = 1;
            else
                maxLossList(j) = 0;
            end
            if maxLossList(j) == outl(j)
                right = right + 1;
                ges = ges +1;
            else
                ges = ges +1;
            end
        end
    end
    acc = right/ges;
    %if isnan(acc)
    %    acc = 1;
    %end
    diag(i) = acc;
end
diagnew = imresize(diag,0.333333, 'nearest')*100

length = 50
x = linspace(0.5,2.0,length)

diagnewpoly = diagnew(~isnan(diagnew));
xpoly = x(~isnan(diagnew));

p = polyfit(xpoly,diagnewpoly,4)
y = polyval(p,x)

plot(x,diagnew, 'b*')
hold on
plot(x,y)
grid on


