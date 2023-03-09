""" =====================================================================================================================
* FeetStat.py
* ---------------------------------------------------------------------------------------------------------------------
* Implementation of the measure extraction code from the point cloud (that we got with Scandy Pro)
* @author Benjamin Tibi 
* =====================================================================================================================
"""


from sklearn_extra.cluster import CommonNNClustering
from scipy.signal import argrelextrema
from sklearn.decomposition import PCA
from sklearn import linear_model
from plyfile import PlyData

import matplotlib.pyplot as plt
import pickle, os, cv2
import open3d as o3d
import numpy as np 



class FeetStat : 
    def __init__(self, name) :
        # Open scan 
        self.name = name 
        self.Read()
        # Tools for analyzis 
        self.ransac = linear_model.RANSACRegressor(residual_threshold=0.45)
        self.clusterKNN = CommonNNClustering(eps=2, min_samples=3)
        self.pca_analyze = PCA(2)

    def Read(self) : 
        """
        Read the point cloud from the app (byte format) and save a copy in the cache
        Format shoud be     X Y Z [R G B]
        """
        if not ('_' + self.name ) in os.listdir('./SCAN_Cache') : 
            print('Opening Scan from the file ... ')
            # Open 
            self.plydata = PlyData.read('./SCAN/' + self.name)
            self.Format()
            # Put in cache 
            pickle.dump(np.concatenate((self.xyz, self.rgb), axis=1), open('./SCAN_Cache/_' + self.name , 'wb'))
        else : 
            print('Opening Scan from the cache ... ')
            # Open 
            res = pickle.load(open('SCAN_Cache/_' + self.name , 'rb'))
            self.xyz = res[:, :3]
            if np.shape(res)[1] == 6 : self.rgb = res[:, 3:]
            else : self.rgb = 0*self.xyz 


    def Format(self) : 
        """
        Format data from the PLY scan file 
        Structure should be like      X Y Z [R G B]
        """
        # CAST     X Y Z
        X = self.plydata.elements[0].data['x']
        Y = self.plydata.elements[0].data['y']
        Z = self.plydata.elements[0].data['z']
        # CAST     [R G B]
        try : 
            R = self.plydata.elements[0].data['red'  ]
            G = self.plydata.elements[0].data['green']
            B = self.plydata.elements[0].data['blue' ]
        except : pass
        # Fill lists    ->  self.xyz
        self.xyz, self.rgb = [], []
        for indice in range(len(X)) : 
            self.xyz.append([X[indice], Y[indice], Z[indice]])
        self.xyz = np.array(self.xyz)
        # Fill lists    ->  self.rgb
        try : 
            for indice in range(len(X)) : 
                self.rgb.append([R[indice], G[indice], B[indice]])
        except : self.rgb = 0*self.xyz.copy()
        self.rgb = np.array(self.rgb)


    def Remove_Ground_RANSAC(self) : 
        """
        Extract the GROUND from the Point Cloud with RANSAC fitting to orient the referential
        """
        # Ransac 
        self.ransac.fit(self.xyz[:, :2], self.xyz[:, 2])
        outlier_mask = np.logical_not(self.ransac.inlier_mask_)
        # Remove ground 
        self.xyz = self.xyz[outlier_mask, :]
        self.rgb = self.rgb[outlier_mask, :]

    def Align_Repere_Ground(self) : 
        """
        Use the ground to align the referential of the point cloud with change of arithmetic bases
        First vector will be ⊥ to the ground 
        """
        # == Ox (ground normal )
        Ox = np.array([[self.ransac.estimator_.coef_[0]], [self.ransac.estimator_.coef_[1]], [-1]])
        Ox = Ox/np.linalg.norm(Ox)
        # == Oy (perpendicular ground normal)
        Oy = np.array([[1], [1], [1]])
        Oy[0] = -(1/Ox[0])*(Ox[1]*Oy[1] + Ox[2]*Oy[2] )
        Oy = Oy/np.linalg.norm(Oy)
        # == Oz (cross product to get referentiel)
        Oz = np.cross(Ox, Oy, axis=0)
        Oz = Oz/np.linalg.norm(Oz)
        # Transition matrix + change 
        MP = np.concatenate((Ox, Oy, Oz), axis=1)
        self.xyz = np.transpose( np.linalg.inv(MP)@np.transpose(self.xyz)  )
 
    def Filter_Distance(self, Delt=30, h=10) : 
        """
        Remove useless data : 
            - over [Delt cm] of the center of the referential 
            - over [h cm] of the ground 
        """
        # Remove everything after 0.5m
        R = self.xyz[:, 1]**2 + self.xyz[:, 2]**2
        indice = (R < Delt**2) & (self.xyz[:, 0] < h)
        # Clean the point cloud 
        self.xyz = self.xyz[indice, :]
        self.rgb = self.rgb[indice, :]


    def extract_Foot(self) : 
        """
        Find Translation to put the referential in the feet area based on maxima reseach 
        """
        # Discretize space
        grid_size =  0.1
        X, Y = self.xyz[:, 1].copy(), self.xyz[:, 2].copy()
        minXp, minYp = np.min(X), np.min(Y)
        maxXp, maxYp = np.max(X), np.max(Y)
        X, Y = (X - minXp)//grid_size, (Y - minYp)//grid_size

        # Create and complet grid 
        MyGrid = np.zeros(( int( (maxXp - minXp)/grid_size ) ,  int( (maxYp - minYp)/grid_size ) ))
        for ind in range(len(X)) : 
            try : MyGrid[int(X[ind]), int(Y[ind])] += 1
            except : pass

        # Loof for local maxima 
        MyGrid = cv2.blur(MyGrid, (150, 150))
        X_max, Y_max = argrelextrema(MyGrid, np.greater) 
        Z_max = np.array([MyGrid[X_max[i], Y_max[i]] for i in range(len(X_max))])
        indice = (Z_max > np.max(Z_max)/1.5)
        X_max = X_max[indice].astype(float)
        Y_max = Y_max[indice].astype(float)

        # Look for the best Maxima to put the marker  
        R = (X_max - int( (maxXp - minXp)/grid_size )/2 )**2 + (Y_max - int( (maxYp - minYp)/grid_size )/2 )**2 
        omegaX, omegaY = X_max[np.argmin(R)], Y_max[np.argmin(R)]

        # Center the marker 
        self.xyz[:, 1] -= (omegaX*grid_size + minXp)
        self.xyz[:, 2] -= (omegaY*grid_size + minYp)

        # plt.scatter(Y_max, X_max, color='g', marker='x', alpha=0.5)
        # plt.scatter(omegaY, omegaX, color='r', marker='o')
        # plt.imshow( MyGrid/np.max(MyGrid) )
        # plt.show()

    def clusterize(self) : 
        """
        Use KNN algorithm to clusterinze point cloud and keep only the points related to the feet for the mensuration 
        measurments and the PCA
        """
        cobj = self.clusterKNN.fit(self.xyz)
        labels = cobj.labels_
        labelsMax = labels[np.argmax( [np.sum(labels == label) for label in np.unique(labels)] )]
        self.xyz = self.xyz[(labels == labelsMax), :]
        self.rgb = self.rgb[(labels == labelsMax), :]
        

    def display_3D(self) : 
        """
        Display remaining point cloud 
        """
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.xyz)
        pcd.colors = o3d.utility.Vector3dVector(self.rgb.astype(np.float) / 255)
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10, origin=[0, 0, 0])
        o3d.visualization.draw_geometries([pcd, mesh_frame])


    def get2D_feet(self) :
        """
        Extract mensuration of the feet L*l in cm
        """
        # 2D Point cloud 
        X, Y = self.xyz[:, 1].copy(), self.xyz[:, 2].copy()

        # PCA
        POINTS = np.concatenate((X.reshape((-1, 1)), Y.reshape((-1, 1))), axis=1)
        self.pca_analyze.fit(POINTS)
        POINTS = self.pca_analyze.transform(POINTS)
        X, Y = POINTS[:, 0], POINTS[:,1]

        # Get length L
        L = np.max(X) - np.min(X)
        print( f"L : {L} cm")

        # Get width 
        dX = np.linspace(min(X), max(X), 10)
        Ylst = []
        for ind in range(1, len(dX) ): 
            indiceX = (X > dX[ind-1]) & (X < dX[ind])
            Ylst.append(np.max(Y[indiceX]) - np.min(Y[indiceX]))

        l = np.max(Ylst)
        rate = (dX[np.argmax(Ylst)] - min(X))/(np.max(X) - np.min(X))
        print( f"l : {l} cm")
        print( f"rate : {rate} ")

        # Print Figure 
        ABSCX, ABSCY = np.linspace(np.min(X), np.max(X), 100), np.linspace(np.min(Y), np.max(Y), 100)
        plt.plot(ABSCX, 0*ABSCX, color='r')
        plt.plot(ABSCY*0 + min(X) + rate*L,   ABSCY, color='r')
        plt.scatter(X, Y, marker='x', color='b', alpha=0.1)
        plt.title(f"{self.name} ->  L : [{round(L,1)} cm] / l : [{round(l, 1)} cm] / rate : [{round(rate, 1)}]")
        plt.xlim([-20, 20])
        plt.ylim([-20, 20])
        plt.grid(True)
        plt.savefig(f"RESULTS/{self.name[:-4]}.png")
        plt.show()

    def run(self) : 
        """
        Run the entire program to get the mensurations 
        """
        self.display_3D()
        # remove ground 
        self.Remove_Ground_RANSAC()
        # align ground 
        self.Align_Repere_Ground()
        # Extract foot position 
        self.extract_Foot()
        self.Filter_Distance()
        # Clean area
        self.clusterize()
        # Display 3D
        self.display_3D()
        # Get mensuration 
        self.get2D_feet()



    



if __name__ == '__main__' : 
    FeetStat(name='Benjamin2.ply').run()
    
    