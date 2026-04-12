function [ZC,c,l]=microstrip_single(eps_r,w,t,h)

mu0  = 4*pi*1e-7;
eps0 = 8.8542e-12;


% ------------------------------------------------------------------
%%  for w/h1

if w/h>1
    eps_eff = ((eps_r+1)/2)+((eps_r-1)/2)*(1/sqrt((1+(12*h/w))));  
else
    eps_eff = ((eps_r+1)/2)+((eps_r-1)/2)*((1/sqrt((1+(12*h/w))))+0.04*(1-w/h)^2);  
end

dw = t/pi*log((4*exp(1)/(((t/h)^2)+(((1/pi)/(w/t+1.1))^2))));            
dww = dw*((1+1/eps_eff)/2);
ww = w+dww;
    
d = sqrt( (((14+(8/eps_eff))/11)^2)*((4*h/ww)^2) + ((1+(1/eps_eff))/2)*pi^2);

b = ( ((14+(8/eps_eff))/11)*(4*h/ww) + d);

a = 1+(4*h/ww)*b;

ZC = (120*pi/(2*sqrt(2)*pi*sqrt(eps_r+1)))*log(a);

c = sqrt((mu0*eps0*eps_r)/(ZC^2));

l = (mu0*eps0*eps_r)/c;



end
