import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import tkinter as tk
from tkinter import filedialog
from scipy.spatial import Delaunay
import triangle as tr
from tkinter.simpledialog import askinteger
import scipy.sparse as sp
import scipy.sparse.linalg
import copy
from scipy.sparse.linalg import lsqr

class InteractivePolyline:
    def __init__(self, ax, fig):
        self.ax = ax
        self.canvas = ax.figure.canvas
        self.line = Line2D([], [], marker='o', color='r')
        self.first_contours = []
        self.ax.add_line(self.line)
        self.xs, self.ys = [], []
        self.is_closed = False
        self.selected_index = None
        self.mesh_created = False
        self.handles = []  # Liste des handles sélectionnés
        self.fixed_points = []  # Liste des points fixes sélectionnés
        self.selecting_special_points = False  # Indique si l'utilisateur est en train de sélectionner les points spéciaux
        self.selected_special_point = False #un point spécial va etre modifié
        self.special_points_confirmed = False # les points spéciaux ont tous été placé
        self.points_selected_over = False
        self.fig = fig
        self.special_points_artists = []
        self.num_handles = 0
        self.num_fixed_points = 0
        self.confirm_button_ax = self.fig.add_axes([0.81, 0.05, 0.1, 0.075])
        self.confirm_button = plt.Button(self.confirm_button_ax, 'Confirmer')
        self.confirm_button.on_clicked(self.confirm_finalization)
        self.confirm_button_ax.set_visible(False)
        self.K = []
        self.V = []
        self.A = []
        self.new_handles = []
        self.remove_handles = False
        self.handles_before = []
        self.handle_that_moves_index = None
        self.handles_indices = []
        self.fixed_indices = []
        self.contours_indices_tuple = []
        self.register = True
        self.use_file = True
        self._connect()


    def _connect(self):
        self.cidpress = self.canvas.mpl_connect('button_press_event', self.on_click)
        self.cidmotion = self.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.cidrelease = self.canvas.mpl_connect('button_release_event', self.on_release)

    def ask_for_special_points_count(self):
        self.num_handles = askinteger("Handles", "handles number:", minvalue=0, maxvalue=50)
        self.num_fixed_points = askinteger("Points Fixes", "Fixed points number:", minvalue=0, maxvalue=50)

    def on_click(self, event):
        if event.inaxes != self.ax:
            return

        if event.button == 1:  # Bouton gauche de la souris
            if not self.is_closed:
                self.xs.append(event.xdata)
                self.ys.append(event.ydata)
                if len(self.xs) > 1 and np.hypot(self.xs[0] - self.xs[-1], self.ys[0] - self.ys[-1]) < 0.15:
                    self.is_closed = True
                    self.xs[-1], self.ys[-1] = self.xs[0], self.ys[0]
                    
                    if self.register:

                        with open("presentation.txt", "w") as file :

                            for x,y in zip(self.xs, self.ys):
                                file.write(f'{x} {y}\n')
                self.line.set_data(self.xs, self.ys)
                self.first_contours.append(self.line) 
                self.canvas.draw()
            else:
                closest_index, min_dist = self._get_closest_point_index(event)
                if min_dist < 0.15:
                    self.selected_index = closest_index

        elif event.button == 3 and len(self.xs) > 0 and not self.is_closed:  # Bouton droit de la souris
            self.xs.pop()
            self.ys.pop()
            self.line.set_data(self.xs, self.ys)
            self.canvas.draw()
        

        if self.is_closed and not self.special_points_confirmed:
            if not self.selecting_special_points:
                self.selecting_special_points = True
                print("Veuillez sélectionner les handles et les points fixes.")
            else:
                # Ajouter et dessiner le point cliqué comme handle ou point fixe
                if len(self.handles) < self.num_handles:
                    self.handles.append((event.xdata, event.ydata))
                    artist, = self.ax.plot(event.xdata, event.ydata, marker='o', markersize=10, color='black')
                    self.special_points_artists.append(artist)  # Stocker la référence
                    print("Handle ajouté.")
                elif len(self.fixed_points) < self.num_fixed_points:
                    self.fixed_points.append((event.xdata, event.ydata))
                    artist, = self.ax.plot(event.xdata, event.ydata, marker='o', markersize=10, color='green')
                    self.special_points_artists.append(artist)  # Stocker la référence
                    print("Point fixe ajouté.")
                self.canvas.draw()

        if len(self.handles) == self.num_handles and len(self.fixed_points) == self.num_fixed_points and not self.remove_handles :
            
            self.confirm_button_ax.set_visible(True)
            self.special_points_confirmed = True
            self.canvas.draw()

        if self.is_closed and self.special_points_confirmed and not self.points_selected_over :
            
            for point_list in [self.handles, self.fixed_points]:
                for i, point in enumerate(point_list):
                    if np.hypot(point[0] - event.xdata, point[1] - event.ydata) < 0.15:
                        self.select_special_point = (point_list, i)  # Store which point is selected
                        self.selected_special_point = True
                        return
                    
        if self.remove_handles:
            
            
            for point_list in [self.handles]:
                for i, point in enumerate(point_list):
                    if np.hypot(point[0] - event.xdata, point[1] - event.ydata) < 0.15:
                        self.select_special_point = (point_list, i)  # Store which point is selected
                        self.selected_special_point = True
                        return


        
    def _get_closest_point_index(self, event):
        min_dist = float('inf')
        closest_index = -1
        for i, (x, y) in enumerate(zip(self.xs, self.ys)):
            dist = np.hypot(x - event.xdata, y - event.ydata)
            if dist < min_dist:
                min_dist = dist
                closest_index = i
        return closest_index, min_dist


    def on_motion(self, event):

        if self.remove_handles:

            if self.selected_special_point:
                point_list, i = self.select_special_point
                point_list[i] = (event.xdata, event.ydata)
                self.handles[i] = (event.xdata, event.ydata)
                #self.update_handles_points()
                L = self.compute_laplacian()
                
                self.handle_that_moves_index = find_vertex_indices(self.V, [self.handles_before[i]])
                self.V =copy.deepcopy(update_mesh(self.V, L, self.fixed_indices, self.handle_that_moves_index, self.handles[i]))
                self.update_handles_points()
                for i in range(len(self.handles)):

                    self.handles[i] = self.V[self.handles_indices[i]]

                #self.update_handles_points()
                self.handles_before = copy.deepcopy(self.handles)
                self.redraw_mesh()


        if event.inaxes != self.ax or self.points_selected_over:
            return

        # If a vertex of the polyline is selected, update its position
        if self.selected_index is not None:
            self.xs[self.selected_index] = event.xdata
            self.ys[self.selected_index] = event.ydata

            # Update both ends if the first/last point is selected
            if self.is_closed and (self.selected_index == 0 or self.selected_index == len(self.xs) - 1):
                self.xs[0] = self.xs[-1] = event.xdata
                self.ys[0] = self.ys[-1] = event.ydata

            self.canvas.draw()
            self.line.set_data(self.xs, self.ys)
        
        if self.selected_special_point:
            point_list, i = self.select_special_point
            point_list[i] = (event.xdata, event.ydata)
            self.update_special_points()
        

    def update_special_points(self):
        # Supprimer les markers existants
        
        if hasattr(self, 'special_points_artists'):
            for artist in self.special_points_artists:
                artist.remove()
            del self.special_points_artists[:]  

        # Réinitialiser la liste des artists
        self.special_points_artists = []
        
        for point in self.handles:
            artist, = self.ax.plot(point[0], point[1], marker='o', markersize=10, color='black')
            self.special_points_artists.append(artist)
        for point in self.fixed_points:
            artist, = self.ax.plot(point[0], point[1], marker='o', markersize=10, color='green')
            self.special_points_artists.append(artist)

        
        self.canvas.draw()

    def update_handles_points(self):
    
        # Supprimer les markers existants et réinitialiser la liste
        if hasattr(self, 'special_points_artists'):
            for artist in self.special_points_artists:
                artist.remove()
            del self.special_points_artists[:]

        self.special_points_artists = []


        # Dessiner les nouveaux markers pour les points spéciaux
        for point in self.handles:
            artist, = self.ax.plot(point[0], point[1], marker='o', markersize=10, color='black')
            self.special_points_artists.append(artist)
            

        for point in self.fixed_points:
            artist, = self.ax.plot(point[0], point[1], marker='o', markersize=10, color='green')
            self.special_points_artists.append(artist)

        self.canvas.draw()


    def perform_triangulation(self):
        
        points_contour = np.array(list(zip(self.xs, self.ys[:-1])))
        segments = np.array([[i, (i + 1) % len(points_contour)] for i in range(len(points_contour))]) 

        
        all_points = np.vstack([points_contour, np.array(self.handles), np.array(self.fixed_points)])

        A = dict(vertices=all_points, segments=segments)

        # Triangulation contrainte
        B = tr.triangulate(A, 'pa0.3')

         # Construire la matrice de connectivité K
        self.K = B['triangles']

        # Construire la matrice des coordonnées V
        self.V = B['vertices']
        
        self.mesh_lines = []

        # Dessiner les segments de maillage sur le même axe
        for i, simplex in enumerate(B['triangles']):
            for j in range(3):
                start_point = B['vertices'][simplex[j]]
                end_point = B['vertices'][simplex[(j + 1) % 3]]
                line, = self.ax.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], color='blue')
                self.mesh_lines.append(line)

        self.canvas.draw()

    def on_release(self, event):
        self.selected_index = None
        self.selected_special_point = False

    def load_points(self, points):
        self.xs, self.ys = list(points[:, 0]), list(points[:, 1])
        self.is_closed = True
        self.line.set_data(self.xs, self.ys)
        self.canvas.draw()

    def confirm_finalization(self, event):
        self.points_selected_over = True
        print("Configuration confirmée. Triangulation en cours...")
        self.perform_triangulation()
        print("Triangulation terminée")
        self.confirm_button_ax.set_visible(False)
        self.A = build_adjacency_matrix(self.K)
        self.remove_handles = True
        self.handles_before = copy.deepcopy(self.handles)
        self.handles_indices = find_vertex_indices(self.V, self.handles_before)
        self.fixed_indices = find_vertex_indices(self.V, self.fixed_points)
        self.contours_indices_tuple = self.trouver_segments_contour(self.V)
        if self.line in self.ax.lines:
            self.line.remove()  # Supprime la ligne de l'axe
        self.canvas.draw()  # Met à jour le canvas pour refléter la suppression


    def trouver_segments_contour(self, V):
        seuil_proximite = 0.001  # Seuil de distance pour considérer deux points comme équivalents
        segments_contour = []

        # Parcourir chaque segment de contour
        for i in range(len(self.xs) - 1):
            point_contour_debut = np.array([self.xs[i], self.ys[i]])
            point_contour_fin = np.array([self.xs[i + 1], self.ys[i + 1]])

            # Comparer avec chaque paire de points dans V
            for j in range(len(V)):
                for k in range(j + 1, len(V)):
                    if np.linalg.norm(point_contour_debut - V[j]) < seuil_proximite and \
                    np.linalg.norm(point_contour_fin - V[k]) < seuil_proximite:
                        segments_contour.append((j, k))
                        break  # Arrêter la recherche une fois la correspondance trouvée

        return segments_contour


    def redraw_mesh(self):
        # Supprimer les anciennes lignes du maillage
        for line in self.mesh_lines:
            line.remove()
        self.mesh_lines.clear()

        # Dessiner les nouveaux segments du maillage en utilisant la matrice d'adjacence
        n_vertices = self.V.shape[0]
        for i in range(n_vertices):
            for j in range(i+1, n_vertices):  # Pour éviter de dessiner deux fois le même segment
                if self.A[i, j]:  # Si il y a un segment entre i et j
                    start_point = self.V[i]
                    end_point = self.V[j]
                    if (i, j) in self.contours_indices_tuple or (j, i) in self.contours_indices_tuple:
                        color = 'blue'  # Couleur pour les contours
                    else:
                        color = 'blue'  # Couleur pour les autres segments

                    # Dessiner le segment
                    line, = self.ax.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], color=color)
                    self.mesh_lines.append(line)

        # Redessiner le canvas
        self.canvas.draw()

    def compute_laplacian(self):
        """
        Computes the Laplacian matrix for a given mesh.

        Parameters:
        - K: Connectivity matrix (n_vertices x n_vertices)

        Returns:
        - L: Laplacian matrix
        """
        degrees = np.sum(self.A, axis=1).flatten()
        D = sp.diags(degrees)
        L = D - self.A
        return L
    

def update_mesh(V, L, fixed_indices, handle_index, new_handle_pos):
    """
    Updates the mesh based on the movement of a handle.

    Parameters:
    - V: Original vertex positions (n_vertices x 2)
    - L: Laplacian matrix
    - fixed_indices: Indices of vertices that are fixed
    - handle_index: Index of the handle vertex
    - new_handle_pos: New position of the handle vertex (1x2)

    Returns:
    - new_V: Updated vertex positions (n_vertices x 2)
    """
    n_vertices = V.shape[0]

    # Create a mask for free vertices (not fixed and not the handle)
    free_vertices = np.ones(n_vertices, dtype=bool)
    free_vertices[fixed_indices] = False
    free_vertices[handle_index] = False

    # Prepare the RHS of the linear system
    b = np.zeros((n_vertices, 2))
    b[free_vertices, :] = L[free_vertices, :] @ V
    b[fixed_indices, :] = V[fixed_indices, :]
    b[handle_index, :] = new_handle_pos

    # Modify the Laplacian to incorporate fixed vertices and the handle
    L_modifie = L.copy()
    L_modifie[fixed_indices, :] = 0
    L_modifie[handle_index, :] = 0
    L_modifie[fixed_indices, fixed_indices] = 1
    L_modifie[handle_index, handle_index] = 1

    # Solve the linear system for the new vertex positions
    new_V = np.vstack([scipy.sparse.linalg.lsqr(L_modifie, b[:, dim])[0] for dim in range(2)]).T

    return new_V


def find_vertex_index(V, point, epsilon=0.01):
    """
    Find the index of a given point in the vertex matrix V.

    Parameters:
    - V: Vertex matrix (n_vertices x 2)
    - point: Tuple representing the point to find (x, y)
    - epsilon: Tolerance for considering two points as equal

    Returns:
    - index: The index of the point in V, or None if not found
    """
    for i, vertex in enumerate(V):
        if np.linalg.norm(vertex - np.array(point)) < epsilon:
            return i
    return None

    

def build_adjacency_matrix(K):
    """
    Construit une matrice d'adjacence à partir de la matrice de connectivité K.

    """
    N = K.max() + 1  # Nombre de sommets
    M = np.zeros((N, N), dtype=int)

    for triangle in K:
        for i in range(3):
            for j in range(i + 1, 3):
                M[triangle[i], triangle[j]] = 1
                M[triangle[j], triangle[i]] = 1

    return M

def ask_load_file():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename()
    root.destroy()
    return file_path


def read_polyline(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    points = [tuple(map(float, line.strip().split(' '))) for line in lines]
    return np.array(points)


def find_vertex_indices(V, points, epsilon=0.01):
    """
    Find the indices of a list of points in the vertex matrix V.

    Parameters:
    - V: Vertex matrix (n_vertices x 2)
    - points: List of tuples representing the points to find [(x1, y1), (x2, y2), ...]
    - epsilon: Tolerance for considering two points as equal

    Returns:
    - indices: List of indices for each point in points, None for not found points
    """
    indices = []
    for point in points:
        index = None
        for i, vertex in enumerate(V):
            if np.linalg.norm(vertex - np.array(point)) < epsilon:
                index = i
                break
        indices.append(index)
    return indices


def find_moved_vertex_index(self_handles, handles, epsilon=0.01):
    """
    Find the index of the vertex that moved between two lists of tuples.

    Parameters:
    - self_handles: List of tuples representing the original points [(x1, y1), (x2, y2), ...]
    - handles: List of tuples representing the updated points [(x1', y1'), (x2', y2'), ...]
    - epsilon: Tolerance for considering whether a point has moved

    Returns:
    - index: The index of the moved vertex, or None if no movement detected
    """
    for i, (original, updated) in enumerate(zip(self_handles, handles)):
        if np.linalg.norm(np.array(original) - np.array(updated)) > epsilon:
            return i
    return None


user_choice = input("Upload your own file ? (y/n): ")
fig, ax = plt.subplots(figsize=(15, 10))
ax.set_title('Polyline Interactive')
ax.set_xlim(0, 20)
ax.set_ylim(0, 20)
ip = InteractivePolyline(ax, fig)
ip.ask_for_special_points_count()

if user_choice.lower() == 'y':
    file_path = ask_load_file()
    if file_path:
        polyline_data = read_polyline(file_path)
        ip.load_points(polyline_data)
       

plt.show()







