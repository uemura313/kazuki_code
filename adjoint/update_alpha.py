import numpy as np
from mpi4py import MPI
import dedalus.public as d3
import config
from trans import scatter_from_zero

def load_field_safely(filename, fields, comm):
    """ファイルの存在を全コアで共有し、デッドロックを防ぐ安全な読み込み関数"""
    rank = comm.rank

    file_exists = False
    global_data = None
    if rank == 0:
        try:
            global_data = np.load(filename)
            file_exists = True
            print(f"  Loaded state from: {filename}")
        except FileNotFoundError:
            print(f"  [ERROR] Could not find {filename}.")
            file_exists = False

    # ファイルの有無を全コアにブロードキャスト（デッドロック防止）
    file_exists = comm.bcast(file_exists, root=0)
    if not file_exists:
        return False

    offset = 0
    for field in fields:
        local_size = field['c'].size
        global_size = comm.allreduce(local_size, op=MPI.SUM)

        if rank == 0:
            field_global = global_data[offset : offset + global_size]
            offset += global_size
        else:
            field_global = None

        field_local = scatter_from_zero(field_global, local_size, comm)

        chunk = field_local.reshape(field['c'].shape)
        if field['c'].dtype == np.float64:
            np.copyto(field['c'], chunk.real)
        else:
            np.copyto(field['c'], chunk)
    return True

def update_alpha_step(step_size=0.1):
    """
    1ステップ分の感度計算と alpha の更新を行い、感度の最大絶対値を返す関数。
    """
    comm = MPI.COMM_WORLD
    rank = comm.rank

    Lx, Lz = config.Lx, config.Lz
    Nx, Nz = config.Nx, config.Nz

    coords = d3.CartesianCoordinates('x', 'z')
    dist = d3.Distributor(coords, dtype=np.float64)
    xbasis = d3.RealFourier(coords['x'], size=Nx, bounds=(0, Lx), dealias=3/2)
    zbasis = d3.ChebyshevT(coords['z'], size=Nz, bounds=(0, Lz), dealias=3/2)

    # ベースフロー変数定義
    p_base = dist.Field(name='p_base', bases=(xbasis,zbasis))
    b_base = dist.Field(name='b_base', bases=(xbasis,zbasis))
    u_base = dist.VectorField(coords, name='u_base', bases=(xbasis,zbasis))
    tau_p_b = dist.Field(name='tau_p_b')
    tau_b1_b = dist.Field(name='tau_b1_b', bases=xbasis)
    tau_b2_b = dist.Field(name='tau_b2_b', bases=xbasis)
    tau_u1_b = dist.VectorField(coords, name='tau_u1_b', bases=xbasis)
    tau_u2_b = dist.VectorField(coords, name='tau_u2_b', bases=xbasis)
    base_fields = [p_base, b_base, u_base, tau_p_b, tau_b1_b, tau_b2_b, tau_u1_b, tau_u2_b]

    # スター変数定義
    p_star = dist.Field(name='p_star', bases=(xbasis,zbasis))
    b_star = dist.Field(name='b_star', bases=(xbasis,zbasis))
    u_star = dist.VectorField(coords, name='u_star', bases=(xbasis,zbasis))
    tau_p_s = dist.Field(name='tau_p_s')
    tau_b1_s = dist.Field(name='tau_b1_s', bases=xbasis)
    tau_b2_s = dist.Field(name='tau_b2_s', bases=xbasis)
    tau_u1_s = dist.VectorField(coords, name='tau_u1_s', bases=xbasis)
    tau_u2_s = dist.VectorField(coords, name='tau_u2_s', bases=xbasis)
    star_fields = [p_star, b_star, u_star, tau_p_s, tau_b1_s, tau_b2_s, tau_u1_s, tau_u2_s]

    alpha = dist.Field(name='alpha', bases=(xbasis, zbasis))

    steady_file = f"steady_state_Ra_{config.Ra:.2e}.npy"
    star_file = f"adjoint_star_Ra_{config.Ra:.2e}.npy"
    alpha_file = "current_alpha.npy"

    if rank == 0:
        print(f"==================================================")
        print(f" Calculating Sensitivity & Updating Alpha")
        print(f"==================================================")

    # 1. データの安全な読み込み
    if not load_field_safely(steady_file, base_fields, comm): return -1.0
    if not load_field_safely(star_file, star_fields, comm): return -1.0

    # 2. alphaの安全な読み込みまたは初期化（全コアで同期）
    local_size_alpha = alpha['c'].size
    global_size_alpha = comm.allreduce(local_size_alpha, op=MPI.SUM)

    file_exists = False
    global_alpha = None
    if rank == 0:
        try:
            global_alpha = np.load(alpha_file)
            file_exists = True
            print(f"  Loaded alpha from: {alpha_file}")
        except FileNotFoundError:
            file_exists = False

    file_exists = comm.bcast(file_exists, root=0)

    if not file_exists:
        if rank == 0:
            print(f"  No {alpha_file} found. Initializing alpha = 1.0")
        global_alpha = np.ones(global_size_alpha, dtype=np.float64)

    alpha_local = scatter_from_zero(global_alpha, local_size_alpha, comm)
    np.copyto(alpha['c'], alpha_local.reshape(alpha['c'].shape).real)

    if rank == 0:
        print("  -> Data loaded. Calculating sensitivity...")

    # 3. 感度の計算
    u_b_g = u_base['g']
    u_s_g = u_star['g']
    alpha_g = alpha['g']

    dot_product = u_b_g[0] * u_s_g[0] + u_b_g[1] * u_s_g[1]
    sensitivity = alpha_g * dot_product

    local_max_sens = np.max(np.abs(sensitivity))
    global_max_sens = comm.allreduce(local_max_sens, op=MPI.MAX)

    if rank == 0:
        print(f"  -> Max Sensitivity = {global_max_sens:.4e}. Updating alpha...")

    # 4. alphaの更新と非負制約
    alpha['g'] = alpha_g - step_size * sensitivity
    alpha['g'] = np.clip(alpha['g'], 0.0, None)

    # 5. 更新された alpha の保存
    alpha_local_flat = alpha['c'].flatten()
    gathered_list = comm.gather(alpha_local_flat, root=0)

    if rank == 0:
        new_global_alpha = np.concatenate(gathered_list)
        np.save(alpha_file, new_global_alpha)
        print(f"  [Update] step_size={step_size:.3f} | Max Sensitivity = {global_max_sens:.4e}")
        print(f"  Updated alpha saved to {alpha_file}")
        print(f"==================================================")

    return global_max_sens

if __name__ == "__main__":
    update_alpha_step(step_size=0.1)
