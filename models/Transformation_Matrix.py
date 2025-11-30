import torch
import torch.nn as nn

def compute_polyhedron_buffers(V_tensor, F_tensor, device):
    """
    Computes the transformation matrix (T_matrix) and the vertex coordinates
    transformed into the face-fixed frame (Face_Vertices_face).
    
    Args:
        V_tensor (torch.Tensor): Body-fixed vertex coordinates (N_V, 3).
        F_tensor (torch.Tensor): Face indices (N_F, 3) (already 0-indexed).
        device (torch.device): Target device (e.g., 'cuda').
        
    Returns:
        T_matrix (torch.Tensor): The Transformation Matrix for all faces (N_F, 3, 3).
        Face_Vertices_face (torch.Tensor): Transformed vertex coords (N_F, 3, 3).
    """
    
    # 1. Advanced Indexing: Extract body vertex coordinates for all faces
    # Shape: (N_F, 3 vertices, 3 coords)
    F_tensor = F_tensor - 1
    Face_Vertices_body = V_tensor[F_tensor.long()].to(device) 
    
    # 2. Calculate Edge Vectors (R12 = P2 - P1, R23 = P3 - P2)
    R12 = Face_Vertices_body[:, 1, :] - Face_Vertices_body[:, 0, :]
    R23 = Face_Vertices_body[:, 2, :] - Face_Vertices_body[:, 1, :]
    
    # 3. Calculate Unit Vectors (X_cap, Y_cap, Z_cap) - The face-fixed frame axes
    
    # Z_cap (Face Normal): Normalized Cross Product (R12 x R23)
    nf_raw = torch.cross(R12, R23, dim=1)
    nf_norm = torch.norm(nf_raw, dim=1, keepdim=True)
    Z_cap = nf_raw / nf_norm
    
    # Enforce Outward Normals
    # Calculate Face Centroids in Body Frame
    Face_Centers = torch.mean(Face_Vertices_body, dim=1) # (N_F, 3)
    # Body Centroid (approximate as mean of vertices)
    Body_Center = torch.mean(V_tensor, dim=0, keepdim=True).to(device)
    # Vector from Body Center to Face Center
    Radial_Vec = Face_Centers - Body_Center
    # Check alignment
    dot_prod = torch.sum(Z_cap * Radial_Vec, dim=1, keepdim=True)
    # Create flip mask (1 if aligned, -1 if opposed)
    # We want dot_prod > 0. If < 0, flip.
    flip_mask = torch.sign(dot_prod)
    # Handle zero case (rare) - default to 1
    flip_mask[flip_mask == 0] = 1
    
    Z_cap = Z_cap * flip_mask
    
    # X_cap (First Edge): Normalized R12
    R12_norm = torch.norm(R12, dim=1, keepdim=True)
    X_cap = R12 / R12_norm
    
    # Y_cap (Orthogonal): Z_cap x X_cap
    Y_cap = torch.cross(Z_cap, X_cap, dim=1) 
    
    # 4. Assemble Transformation Matrix T_matrix (Rows are X, Y, Z caps)
    T_matrix = torch.stack((X_cap, Y_cap, Z_cap), dim=1).to(device) # Shape: (N_F, 3, 3)
    
    # 5. Calculate Face-Fixed Vertices (TrCo equivalent)
    # P_face = T_matrix @ P_body_col. This needs a batch matrix multiplication.
    
    # Extract body vertices as column vectors for batch matrix multiplication
    P1_col = Face_Vertices_body[:, 0, :].unsqueeze(2) # (N_F, 3, 1)
    P2_col = Face_Vertices_body[:, 1, :].unsqueeze(2)
    P3_col = Face_Vertices_body[:, 2, :].unsqueeze(2)
    
    # Apply transformation: (N_F, 3, 3) @ (N_F, 3, 1) -> (N_F, 3, 1)
    V1_face_col = T_matrix @ P1_col
    V2_face_col = T_matrix @ P2_col
    V3_face_col = T_matrix @ P3_col
    
    # Concatenate the resulting transformed vertex vectors (as column stacks)
    Face_Vertices_face = torch.cat([V1_face_col, V2_face_col, V3_face_col], dim=2).to(device) 
    # Final Shape: (N_F, 3 coords, 3 vertices)
    
    return T_matrix, Face_Vertices_face

def __init__():
    pass