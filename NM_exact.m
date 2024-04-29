% Yuhang Li, 000981323
clc;
clear all;
close all;

syms x y;

f = 2 * x^2 + 2 * x * y + y^2 - x - y;
x0 = [1 9];
[x_best,f_best] = Newton(f,x0,[x y]);



function [x_best,f_best] = Newton(f,x0,x,epsilon)

format long;
if nargin == 3  
    epsilon = 1.0e-6;
end

x0 = transpose(x0);
x = transpose(x);
g1f = jacobian(f,x);

 
g2f = jacobian(g1f,x);

grad_fxk = 1;
k = 0;
xk = x0;


while norm(grad_fxk) > epsilon  


    grad_fxk  = subs(g1f,x,xk);
    grad2_fxk = subs(g2f,x,xk);
    pk = -inv(grad2_fxk)*transpose(grad_fxk);  
    pk = double(pk);
    xk_next = xk + pk; 
    xk = xk_next;
    k = k + 1;
    f_1 = subs(f,x,xk);

    fprintf(['iteration # %d  residual %.20f point (x,y) = (%f,%f)' ...
        'ext value f(x,y) = %.20f\n'],...
        k,vpa(norm(grad_fxk)),xk(1),xk(2),vpa(f_1));
end

x_best = xk_next;
f_best = subs(f,x,x_best);
end