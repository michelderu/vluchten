#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Alejandro Murrieta (I just copied it from an old MATLAB file)
All credit goes to RMIT in Australia
My work was mostly to adapt it to python.

Have Fun
"""
    
def v_direct(coord1,coord2,maxIter=200,tol=10**-12):
    if coord1 == coord2:
        m=0
        az12=0
        return m, az12
        
    # Tolerance (tol) in meters
    from math import atan
    from math import atan2
    from math import cos
    from math import radians
    from math import sin
    from math import sqrt
    from math import tan
    from math import pi
    # Test Variables
    # maxIter=200
    # tol=10**-12
    # coord1 = (42.3541165, -71.0693514)
    # coord2 = (40.7791472, -73.9680804)
    
    #--- CONSTANTS ------------------------------------+
    
    a=6378137.0                             # radius at equator in meters (WGS-84)
    f=1/298.257223563                       # flattening of the ellipsoid (WGS-84)
    b=(1-f)*a
    
    phi_1,L_1,=coord1                       # (lat=L_?,lon=phi_?)
    phi_2,L_2,=coord2                  
    
    u_1=atan((1-f)*tan(radians(phi_1))) # u is psi
    u_2=atan((1-f)*tan(radians(phi_2)))
    
    L=radians(L_2-L_1)
    
    Lambda=L                                # set initial value of lambda to L
    
    sin_u1=sin(u_1)
    cos_u1=cos(u_1)
    sin_u2=sin(u_2)
    cos_u2=cos(u_2)
    
    #--- BEGIN ITERATIONS -----------------------------+
    iters=0
    for i in range(0,maxIter):
        iters+=1
        
        cos_lambda=cos(Lambda)
        sin_lambda=sin(Lambda)
        sin_sigma=sqrt((cos_u2*sin(Lambda))**2+(cos_u1*sin_u2-sin_u1*cos_u2*cos_lambda)**2)
        cos_sigma=sin_u1*sin_u2+cos_u1*cos_u2*cos_lambda
        sigma=atan2(sin_sigma,cos_sigma)
        sin_alpha=(cos_u1*cos_u2*sin_lambda)/sin_sigma
        cos_sq_alpha=1-sin_alpha**2
        cos2_sigma_m=cos_sigma-((2*sin_u1*sin_u2)/cos_sq_alpha)
        C=(f/16)*cos_sq_alpha*(4+f*(4-3*cos_sq_alpha))
        Lambda_prev=Lambda
        Lambda=L+(1-C)*f*sin_alpha*(sigma+C*sin_sigma*(cos2_sigma_m+C*cos_sigma*(-1+2*cos2_sigma_m**2)))
    
        # successful convergence
        diff=abs(Lambda_prev-Lambda)
        if diff<=tol:
            break
        
    u_sq=cos_sq_alpha*((a**2-b**2)/b**2)
    A=1+(u_sq/16384)*(4096+u_sq*(-768+u_sq*(320-175*u_sq)))
    B=(u_sq/1024)*(256+u_sq*(-128+u_sq*(74-47*u_sq)))
    delta_sig=B*sin_sigma*(cos2_sigma_m+0.25*B*(cos_sigma*(-1+2*cos2_sigma_m**2)-(1/6)*B*cos2_sigma_m*(-3+4*sin_sigma**2)*(-3+4*cos2_sigma_m**2)))
    
    m=b*A*(sigma-delta_sig)                 # output distance in meters     \\
    y_amm = cos_u2* sin_lambda
    x_amm = (cos_u1 * sin_u2) - (sin_u1*cos_u2*cos_lambda)
    alpha1_amm = atan2(y_amm,x_amm)
    if alpha1_amm < 0:
        alpha1_amm = alpha1_amm +2*pi
    az12 = alpha1_amm*(180/pi);
    return m#, az12
    
def v_inverse(lat1,lon1,az12,s):
    from math import pi
    from math import sin
    from math import cos
    from math import tan
    from math import atan
    from math import atan2
    from math import acos
    from math import sqrt
    
    # Define some constants
    d2r = 180/pi;
    twopi = 2*pi;
    pion2 = pi/2;
    # Set defining ellipsoid parameters
    a = 6378137; # GRS80
    flat = 298.257222101;

    # Compute derived ellipsoid constants
    f = 1/flat;
    b = a*(1-f);
    e2 = f*(2-f);
    ep2 = e2/(1-e2);
    #---------------------------------------
    # latitude and longitude of P1 (degrees)
    #---------------------------------------
    # lat and lon of P1 (radians)
    phi1 = lat1/d2r;
    lambda1 = lon1/d2r;
    #------------------------------------
    # azimuth of geodesic P1-P2 (degrees)
    #------------------------------------
    #az12 = 1 + 43/60 + 25.876544/3600;
    #az12=294 +38/60+ 59.528610/3600;
    
    # azimuth of geodesic P1-P2 (radians)
    alpha1 = az12/d2r;
    # sine and cosine of azimuth P1-P2
    sin_alpha1 = sin(alpha1);
    cos_alpha1 = cos(alpha1);
    #------------------
    # geodesic distance
    #------------------
    #s = 3692399.836991*0.75;
    # [1] Compute parametric latitude psi1 of P1
    psi1 = atan((1-f)*tan(phi1));
    # [2] Compute parametric latitude of vertex
    psi0 = acos(cos(psi1)*sin_alpha1);

# [3] Compute geodesic constant u2 (u-squared)
    u2 = ep2*(sin(psi0)**2);
# [4] Compute angular distance sigma1 on the auxiliary sphere from equator
# to P1'
    sigma1 = atan2(tan(psi1),cos_alpha1);
# [5] Compute the sine of the azimuth of the geodesic at the equator
    sin_alphaE = cos(psi0);
# [6] Compute Vincenty's constants A and B
    A = 1 + u2/16384*(4096 + u2*(-768 + u2*(320-175*u2)));
    B = u2/1024*(256 + u2*(-128 + u2*(74-47*u2)));
    # [7] Compute sigma by iteration
    sigma = s/(b*A);
    iterat = 1;
    while 1:
        two_sigma_m = 2*sigma1 + sigma;
        s1 = sin(sigma);
        s2 = s1*s1;
        c1 = cos(sigma);
        c1_2m = cos(two_sigma_m);
        c2_2m = c1_2m*c1_2m;
        t1 = 2*c2_2m-1;
        t2 = -3+4*s2;
        t3 = -3+4*c2_2m;
        delta_sigma = B*s1*(c1_2m+B/4*(c1*t1-B/6*c1_2m*t2*t3));
        sigma_new = s/(b*A)+delta_sigma;
        if abs(sigma_new-sigma) < 1e-12 :
            break
        sigma = sigma_new;
        iterat = iterat + 1;

    s1 = sin(sigma);
    c1 = cos(sigma);
 # [8] Compute latitude of P2
    y = sin(psi1)*c1+cos(psi1)*s1*cos_alpha1;
    x = (1-f)*sqrt(sin_alphaE**2+(sin(psi1)*s1-cos(psi1)*c1*cos_alpha1)**2);
    phi2 = atan2(y,x);
    lat2 = phi2*d2r;
# [9] Compute longitude difference domega on the auxiliary sphere
    y = s1*sin_alpha1;
    x = cos(psi1)*c1-sin(psi1)*s1*cos_alpha1;
    domega = atan2(y,x);
# [10] Compute Vincenty's constant C
    x = 1-sin_alphaE**2;
    C = f/16*x*(4+f*(4-3*x));
    # [11] Compute longitude difference on ellipsoid
    two_sigma_m = 2*sigma1 + sigma;
    c1_2m = cos(two_sigma_m);
    c2_2m = c1_2m*c1_2m;
    dlambda = domega-(1-C)*f*sin_alphaE*(sigma+C*s1*(c1_2m+C*c1*(-1+2*c2_2m)));
    dlon = dlambda*d2r;
    lon2 = lon1+dlon;
    # [12] Compute azimuth alpha2
    y = sin_alphaE;
    x = cos(psi1)*c1*cos_alpha1-sin(psi1)*s1;
    alpha2 = atan2(y,x);
    # [13] Compute reverse azimuth az21
    az21 = alpha2*d2r + 180;  
    if az21 > 360 :
        az21 = az21-360;
        
    return lat2,lon2
    
## TEST CODE BEGINS HERE
 
#
#    
#boston = (42.3541165, -71.0693514)
#newyork = (40.7791472, -73.9680804)
# 
#distance,azimuth = v_direct(boston, newyork)  # Meteres, degrees
#print(distance)  # 298396.05747326626 mts
#print(azimuth)   # 235.0838926191198  degrees
#
## Define a waypoint
## vinc_in(lat, lon, alpha12, s ) 
#
#lat2,lon2 = inverse(boston[0],boston[1],azimuth,distance)
#print(lat2)     # 40.77914719998252
#print(lon2)     # -73.96808039994268

