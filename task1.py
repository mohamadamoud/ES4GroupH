import numpy as np

from skimage.transform import resize
import scipy.misc
import matplotlib.pyplot as plt

#Get Values from file
values = np.loadtxt(open("pca_dataset.txt", "rb"), delimiter=" ", skiprows=0)


x = values[:,0]
y = values[:,1]

rows = len(values)
columns = len(values[0])


#Centre the matrix
M = np.zeros([rows,columns])
M[:,0] = values[:,0] - values[:,0].mean()
M[:,1] = values[:,1] - values[:,1].mean()


#decompose the matrix
U, S, Vt = np.linalg.svd(M,full_matrices=False)
S = np.diag(S)

#energy percentage
xep = 1/S.trace()*S[0][0]
yep = 1/S.trace()*S[1][1]

print(xep)
print(yep)



fig, ax = plt.subplots()

ax.plot([0,Vt[0][0]*xep],[0,Vt[0][1]*xep],color = "blue")

ax.plot([0,Vt[1][0]*yep],[0,Vt[1][1]*yep], color = 'blue')
ax.axis('equal')
ax.scatter(M[:,0],M[:,1], s= 3, c = "black")
ax.grid()




S0 = S[:-1,:-1]
U0 = U[:-1,:-1]


fig0, ax0 = plt.subplots()
P = -(U0 *S0)
ax0.grid()
ax0.scatter(P, np.zeros(len(P)),s=3, c = "black")
ax0.set_xlabel('PC1')


fig1, ax1 = plt.subplots()
ax1.grid()
P = (U @ S)

ax1.scatter(P[:,0],P[:,1],s=3, c = "black")
ax1.set_xlabel('PC1')
ax1.set_ylabel('PC2')
ax1.set_title('a sine wave')


fig2, ax2 = plt.subplots()

img = scipy.misc.face(gray=True)
img = resize(img, (249, 185))
ax2.imshow(img,cmap='gray')
ax2.set_title('Original')



#Centre the matrix
img_centre = np.zeros([249,185])
rows = len(img_centre)
columns = len(img_centre[0])
for i in range(rows):
    img_centre[i,:] = img[i,:] - img[i,:].mean()

fig3, ax3 = plt.subplots()

ax3.imshow(img_centre,cmap='gray')
ax3.set_title('Centred')

#decompose the matrix
U, s, Vt = np.linalg.svd(img_centre,full_matrices=False)
S = np.diag(s)

energies = np.zeros(len(s))
#energy percentages
for i in range(len(s)):
    energies[i] = 1/S.trace()*S[i][i]


fig4, ax4 = plt.subplots(2,2)
pcs = [len(s), 120,50,10]
ctr = 0

for i in range(len(pcs)):
    pc = pcs[i]
    energy = np.sum(energies[:pc])

    U = U[:, :pc]
    S = np.diag(s[:pc])
    Vt = Vt[:pc, :]
    img_new = U @ S @ Vt

    ax4[i//2][i%2].imshow(img_new,cmap='gray')
    ax4[i//2][i%2].set_title("Principal components = " + str(pc) + " Energy %: " + str(round(energy,2)*100) + "%")

fig5, ax5 = plt.subplots()

ax5.bar(range(1,1+len(s)),energies, color = 'darkblue')

e99 = 0
pp = 0
for ii in range(len(energies)):

    if (e99 >= 0.990):
        pc = ii
        break
    e99 += energies[ii]

fig6, ax6 = plt.subplots()
U, s, Vt = np.linalg.svd(img_centre,full_matrices=False)
energy = np.sum(energies[:pc])
UU = U[:, :pc]
SS = np.diag(s[:pc])
VVt = Vt[:pc, :]

img_new = UU @ SS @ VVt

ax6.imshow(img_new,cmap='gray')
ax6.set_title("Principal components = " + str(pc) + " Energy %: " + str(round(energy,3)*100) + "%")


#Part3
values = np.loadtxt(open("data_DMAP_PCA_vadere.txt", "rb"), delimiter=" ", skiprows=0)



#Centre the matrix
rows = len(values)
columns = len(values[0])

M_centre = np.zeros([rows,columns])

for i in range(rows):
    M_centre[i,:] = values[i,:] - values[i,:].mean()

#decompose the matrix
U, s, Vt = np.linalg.svd(M_centre,full_matrices=False)
S = np.diag(s)

energies = np.zeros(len(s))
#energy percentages
for i in range(len(s)):
    energies[i] = 1/S.trace()*S[i][i]

fig7, ax7 = plt.subplots()
ax7.bar(range(1,1+len(s)),energies,color = 'maroon')

fig9, ax9 = plt.subplots()

ax9.scatter(values[:,0],values[:,1],s= 3)
ax9.scatter(values[:,2],values[:,3],s= 3)



fig8, ax8 = plt.subplots()
pc = 2
UU = U[:, :pc]

SS = np.diag(s[:pc])


R = UU @ SS

ax8.scatter(R[:,0],R[:,1], s= 3, c = "black")
ax8.axis('equal')
ax8.set_xlabel('PC1')
ax8.set_ylabel('PC2')
ax8.set_title("Energy = " + str(100*(energies[0]+energies[1])) + " %")

e90 = 0
pp = 0
for ii in range(len(energies)):

    if (e90 >= 0.90):
        pc = ii
        break
    e90 += energies[ii]

print("Principal Components Needed to get to > 90%: " + str(pc))
print("At PC = " + str(pc) + " energy = " + str(e90*100) + "%")


plt.show()
