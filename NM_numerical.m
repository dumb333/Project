% Yuhang Li, 000981323

function [x_best, f_best] = OptimizeFunction(f, x0)
    syms x y;
    
    if nargin < 2
        % Default initial guess if x0 is not provided
        x0 = [1, 9];
    end

    % Define the function to optimize if it's not provided
    if nargin < 1
        f = 2 * x^2 + 2 * x * y + y^2 - x - y; % Example function
    end
    
    % Call the Newton optimization method
    [x_best, f_best] = Newton(f, x0, [x y]);
end

% Include the rest of the helper functions and the Newton method here
function varargout=Jacobi(f,varargin)
    [x,f]=fx(f);
    n=nargin(f); 
    df=[];
    for i =1:n
        % Step size to 0.1
        df1 = diff(f,x(i))/0.1;
        df = [df,df1];
    end

    varargout{1}=df;
    varargout{2}=matlabFunction(df);
    for i=1:length(x)
        s{i}=char(x(i));
    end
    varargout{3}=s;
    if ~isempty(varargin)
        varargout{4}=Jx(df,s,varargin{1});
    end
end

function [x,f]=fx(f)
    if  ~isa(f,'sym')  
        if iscolumn(f)
            f=str2sym(f);
        else
            f=str2sym(f');
        end                 
    end 
    x=symvar(f);             
    f=matlabFunction(f);
end

function Jk=Jx(J,x,x0)
    n=nargin(matlabFunction(J));
    if n==0
        Jk=double(J);
    else
        a=symvar(J);
        for i=1:length(a)
            s=char(a(i));
            idx(i) = find(strcmp(x,s));
        end
        Jk = subs(J,a,x0(idx));
        Jk = double(Jk);
    end   
end

function [x_best, f_best] = Newton(f, x0, x, epsilon)
    format long;
    if nargin == 3
        epsilon = 1.0e-6;
    end
    x0 = transpose(x0);
    x = transpose(x);
    g1f = Jacobi(f);
    g2f = Jacobi(g1f);

    grad_fxk = 1;
    k = 0;
    xk = x0;

    while norm(grad_fxk) > epsilon
        grad_fxk  = subs(g1f,x,xk);
        grad2_fxk = subs(g2f,x,xk);
        pk = -inv(reshape(grad2_fxk,2,2)) * transpose(grad_fxk); % Step size
        pk = double(pk);
        xk_next = xk + pk; 
        xk = xk_next;
        k = k + 1;
        f_1 = subs(f,x,xk);

        fprintf(['iteration #%d residual %.20f point (x,y) = (%f,%f) ' ...
        'ext value f(x,y) = %.20f\n'], k, vpa(norm(grad_fxk)), xk(1), xk(2), vpa(f_1));
    end

    x_best = xk_next;
    f_best = subs(f,x,x_best);
end

% Example usage:
% [x_optimized, f_optimized] = OptimizeFunction();
