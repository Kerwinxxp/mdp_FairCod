PRIVACY_MODE_FEE = False         # 是否对配送费加噪
PRIVACY_MODE_TIME = False    # 是否对订单时间加噪
NOISE_EPSILON = [1, 3, 5,100]      # 配送费加噪的 ε 参数
SENSITIVITY = 1              # 配送费噪声的灵敏度

TIME_NOISE_EPSILON = [0.5, 1, 1.5]  # 订单时间加噪的 ε 参数（可单独设置）
TIME_SENSITIVITY = 1         # 订单时间噪声的灵敏度
SEED = 42                  
MAX_DAY = 30               
MAX_TIME = 179              
MATRIX_X = 10
MATRIX_Y = 10
MEMORY_SIZE = 100000
BATCH_SIZE = 50
