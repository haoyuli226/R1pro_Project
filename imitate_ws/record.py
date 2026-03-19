import os
import subprocess
import sys
import time
import re
import signal

# ================= 核心配置区 =================
BAG_DIR = "/home/nvidia/imitation_data_pipeline/test_bag"
PREFIX = "pick_cup_"

# 模仿学习：必须绝对齐全的 17 个“黄金话题”
REQUIRED_TOPICS = [
    "/hdas/camera_head/rgb/image_rect_color",
    "/hdas/camera_head/depth/depth_registered",
    "/hdas/camera_wrist_left/color/image_raw/compressed",
    "/hdas/camera_wrist_right/color/image_raw/compressed",
    "/motion_control/pose_ee_arm_left",
    "/motion_control/pose_ee_arm_right",
    "/motion_target/target_pose_arm_left",
    "/motion_target/target_pose_arm_right",
    "/hdas/feedback_gripper_left",
    "/hdas/feedback_gripper_right",
    "/motion_target/target_position_gripper_left",
    "/motion_target/target_position_gripper_right",
    "/hdas/feedback_torso",
    "/motion_target/target_joint_state_torso",
    "/tf",
    "/tf_static",
    "/joint_states"
]

# 金丝雀话题 (Canary Topic): 选择数据量最大、最容易因为 SHM 死锁而卡住的话题进行物理层探活
CANARY_TOPIC = "/hdas/camera_head/rgb/image_rect_color"
# ==============================================

def get_next_bag_name() -> str:
    """
    [引擎 1] 扫描硬盘目录，利用正则自动推算下一个绝对安全的序号
    """
    if not os.path.exists(BAG_DIR):
        os.makedirs(BAG_DIR)
        return f"{PREFIX}001"
    
    existing_dirs = os.listdir(BAG_DIR)
    max_num = 0
    pattern = re.compile(rf"^{PREFIX}(\d+)$")
    
    for d in existing_dirs:
        match = pattern.match(d)
        if match:
            num = int(match.group(1))
            if num > max_num:
                max_num = num
                
    next_num = max_num + 1
    return f"{PREFIX}{next_num:03d}"

def check_data_flow(topic: str, timeout: float = 3.0) -> bool:
    """
    [引擎 2 - 物理层] 强制探活：在限定时间内，真实地从网络中榨取 1 条消息。
    这是拦截 Fast DDS 共享内存死锁的最核心防线。
    """
    print(f"[💧] 正在深入 DDS 传输层，进行 [{topic}] 的物理探活 (限时 {timeout}s)...")
    try:
        # 使用 timeout 限制堵塞，--message-count 1 保证拿到一帧就撤退
        cmd = ['ros2', 'topic', 'echo', topic, '--message-count', '1', '--no-arr']
        # 吞掉繁杂的图像矩阵输出，只关心进程是否能成功返回
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=timeout)
        return True
    except subprocess.TimeoutExpired:
        return False
    except Exception:
        return False

def check_topics():
    """
    [引擎 2 - 拓扑层] 发车前全维度质检：比对节点宣告清单
    """
    print("\n[⏳] 正在扫描当前 ROS 2 拓扑网络中的注册清单...")
    try:
        result = subprocess.run(['ros2', 'topic', 'list'], capture_output=True, text=True, check=True)
        active_topics = set(result.stdout.strip().split('\n'))
    except Exception as e:
        print(f"\n[❌] 致命错误：无法获取拓扑列表，ROS 2 守护进程可能已崩溃: {e}")
        sys.exit(1)

    missing_topics = [t for t in REQUIRED_TOPICS if t not in active_topics]
    
    # 关卡 A：拓扑图完整性检查
    if missing_topics:
        print("\n" + "━"*50)
        print("🚨 【拦截】拓扑层质检失败！发现以下节点未正常发布：")
        for mt in missing_topics:
            print(f"  ❌ {mt}")
        print("━"*50)
        print("💡 建议：检查 R1PROBody.d 或 VR 启动脚本是否运行。")
        sys.exit(0)
        
    # 关卡 B：物理层死锁检查 (金丝雀探活)
    if not check_data_flow(CANARY_TOPIC):
        print("\n" + "━"*50)
        print("🚨 【绝对拦截】遭遇 Fast DDS 共享内存死锁！")
        print(f"❌ 话题 [{CANARY_TOPIC}] 存在于拓扑中，但传输层无任何数据流出！")
        print("━"*50)
        print("\n💡 【一键抢救指令】请立即复制并在终端运行以下命令进行清理：")
        print("\033[93m" + "sudo rm -rf /dev/shm/fastrtps* && ros2 daemon stop && ros2 daemon start" + "\033[0m")
        print("\n🛑 录制已强制阻断，避免产生无效盲包。清理完成后请重试。")
        sys.exit(0)

    print("[✅] 完美！菜单全都在，且物理流探活通过！通信公路状态：\033[92m极佳\033[0m")

def record_bag(bag_name: str):
    """
    [引擎 3] 执行录包进程与安全守护
    """
    bag_path = os.path.join(BAG_DIR, bag_name)
    print(f"\n[🎬] 机器臂感知已就绪！")
    print(f"🎯 目标存储路径: \033[92m{bag_path}\033[0m")
    print("-----------------------------------------------------")
    print("👉 戴上 VR 手柄，随时可以开始抓取动作。")
    print("🛑 动作完成后，在此终端按下【Ctrl + C】即可安全封包。")
    print("-----------------------------------------------------\n")
    
    cmd = ["ros2", "bag", "record", "-o", bag_path] + REQUIRED_TOPICS
    
    # 启动录像子进程
    proc = subprocess.Popen(cmd)
    
    try:
        # 主进程挂起，将终端控制权交给 ros2 bag，同时监听键盘异常
        proc.wait()
    except KeyboardInterrupt:
        print("\n\n[🛡️] 捕捉到终端停止信号 (Ctrl+C)，启动优雅封包程序...")
        # 标准的 POSIX 中断信号，触发 SQLite 数据库安全写入
        proc.send_signal(signal.SIGINT)
        try:
            # 给予底层数据库 5 秒钟的写缓存和锁释放时间
            proc.wait(timeout=5.0)
            print(f"[🎉] 封包成功！黄金数据已保存至: {bag_name}")
            print("👉 提示：直接按方向键 ↑ 然后回车，即可开启下一次录制。")
        except subprocess.TimeoutExpired:
            print("[⚠️] SQLite 封包超时，强制切断进程...")
            proc.kill()

if __name__ == "__main__":
    print("\n" + "█"*45)
    print("   🤖 星海 R1 Pro 工业级数据采集管家")
    print("█"*45)
    
    # 1. 计算编号
    next_bag_name = get_next_bag_name()
    
    # 2. 发车前双重质检 (拓扑层 + 物理层)
    check_topics()
    
    # 3. 安全启动与守护录像进程
    record_bag(next_bag_name)