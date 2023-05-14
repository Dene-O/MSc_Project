import numpy as np


def X_distribution(N_features, N_ranges=0, N_points=0, bounds=1):
    
    if N_ranges == 0 and N_points == 0:
        N_ranges = 5
        N_points = N_features * N_ranges
        
    elif N_features * N_ranges > N_points:
        N_points = N_features * N_ranges
        
    elif int(N_points / N_features) > N_ranges:
        N_ranges = int(N_points / N_features)
        
        

    range_limits = 2 * bounds * np.arange(N_ranges + 1) / N_ranges - bounds
    #print('range_limits',range_limits)
        
    distribution = np.random.uniform(low=-bounds, high=bounds, size=[N_points, N_features])
    
    point_idx = 0
    for feature in range(N_features):
        
        for feature_range in range(N_ranges):
            
            distribution[point_idx, feature] = np.random.uniform(low=range_limits[feature_range], high=range_limits[feature_range+1])
            
            #print('pt', distribution[point_idx, feature],point_idx)
            
            point_idx = point_idx + 1

    return distribution


def L_distribution(N_features, N_points, bounds=1):
    
    range_limits = 2 * bounds * np.arange(N_points + 1) / N_points - bounds
    print('range_limits',range_limits)

    distribution = np.random.uniform(low=-bounds, high=bounds, size=[N_points, N_features])

    for point in range(N_points):
                
        distribution[point, :] = np.random.uniform(low=range_limits[point], high=range_limits[point+1], size = N_features)
            
    for feature in range(1, N_features):            
        
        distribution[:, feature] = Shuffle(distribution[:, feature])            
        
    return distribution
            
def Shuffle(a):
    
    length = a.size
    print(length)
    
    randoms = np.argsort(np.random.uniform(size=length))
    print(randoms)
    
    a_copy = np.empty(length)
    
    for i,r in enumerate(randoms):
        a_copy[i] = a[r]
        
    return a_copy