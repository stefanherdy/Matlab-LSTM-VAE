
x1 = 1:70;
x2 = 1:70;
[X1,X2] = meshgrid(x1,x2);
X = [X1(:) X2(:)];
mu = [22, 33]
%mu1 = [22, 33, 45, 70, 110, 150, 137]
%mu2 = [37, 39, 57, 78, 120, 107, 77]
%sig1 = [37, 20, 20, 13, 47, 9, 23]
%sig2 = [17, 7, 20, 33, 57, 9, 23]
%mu = [22 37],[27 39];
%sigma = [37 0; 0 14],[20 0; 0 7];
sigma = [14 0; 0 32]

matrix= zeros(70, 70)
for i=1:1
    z = mvnpdf(X,[mu1(i) mu2(i)],[sig1(i) 0; 0 sig2(i)]);
    z = mvnpdf(X,mu,sigma);
    z = reshape(z,70,70);
    %z=DataMtx(x1,x2);
    matrix = matrix + z
end

    %z=DataMtx(x1,x2);
    fig1 = figureGen();
    grid on
    h =surf(X1,X2,z)
    set(h,'edgecolor','none', 'linewidth',0.00000001)
    title('Probability density function')
    colormap(flipud(hot));
    zlabel('Probability density')
    xlabel('LS dim. 1')
    ylabel('LS dim. 2')
    %colormap(hot)
    %shading interp
    colorbar; 

