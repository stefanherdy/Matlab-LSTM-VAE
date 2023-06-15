fig1 = figureGen()
plot(maxLossList,outl,'.')
grid on;
xline(1.23, 'r')
axis([0.4 2 -0.1 1.1]);
yticks([0 0.5 1]);
title('Manual Classification in Relation to the maximum Loss')
xlabel('Maximum Loss per Sample')
ylabel('Classification')
yticks([0 1])
yticklabels({'normal','outlier'})

fig2 = figure()
maxLossNew = []
for i=1:length(maxLossList)
    if maxLossList(i)>1.23
        maxLossNew(i)=1;
    else
        maxLossNew(i)=0;
    end
end

C = confusionmat(outl, maxLossNew)

confusionchart(C)