using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using Random = UnityEngine.Random;
using System.Collections.Generic;
using System.IO;

[System.Serializable]
public class RobotConfig
{
    public string robot_name = "X2Ultra";
    public string[] head_keywords;
    public float[] head_kps;
    public float[] head_kds;
    public float[] head_default_angles_rad;

    public string[] arm_keywords;
    public float[] arm_kps;
    public float[] arm_kds;
    public float[] arm_default_angles_rad;

    public string[] waist_keywords;
    public float[] waist_kps;
    public float[] waist_kds;
    public float[] waist_default_angles_rad;

    public string[] leg_keywords;
    public float[] leg_kps;
    public float[] leg_kds;
    public float[] leg_default_angles_rad;

    public float action_scale = 15.0f;
}

public class X02randAgent : Agent
{
    [Header("Basic Config")]
    public bool fixbody = false;
    public bool train = true;
    public string configPath = "x2ultra_stand_default_config.json"; // 确保此文件存在
    
    [Header("Locomotion Config (Gym Style)")]
    public float baseHeightTarget = 0.78f; // 目标躯干高度
    public float stepPeriod = 0.8f;        // 步态周期
    public FootSensor leftFootSensor;      // 必须在Inspector拖入
    public Transform leftFootTransform;    // 必须在Inspector拖入
    public FootSensor rightFootSensor;     // 必须在Inspector拖入
    public Transform rightFootTransform;   // 必须在Inspector拖入

    // --- 内部变量 ---
    RobotConfig robotCfg;
    ArticulationBody[] arts;
    ArticulationBody[] legs = new ArticulationBody[20];
    ArticulationBody[] arms = new ArticulationBody[20];
    ArticulationBody[] waists = new ArticulationBody[3];
    ArticulationBody[] heads = new ArticulationBody[2];
    Transform body;
    
    int legNum = 0, armNum = 0, waistNum = 0, headNum = 0;
    float[] targetDeg = new float[20];
    float[] defaultDeg = new float[20];
    float[] lastActions = new float[20]; // 记录上一帧动作
    float action_scale = 15.0f;

    // --- 状态与指令 ---
    private float episodeTime;
    private float phase;      
    private float phaseLeft;
    private float phaseRight;
    private Vector2 cmdLinVel; // 目标线速度
    private float cmdAngVel;   // 目标角速度

    // --- 域随机化 ---
    [Header("Domain Randomization")]
    public bool randomizeFriction = true;
    public Vector2 frictionRange = new Vector2(0.3f, 1.0f);
    public bool randomizeBaseMass = true;
    public Vector2 baseMassDeltaRange = new Vector2(-1f, 3f);
    public bool pushRobots = true;
    public float pushIntervalSeconds = 5f;
    public float maxPushVelocity = 1.0f;
    
    private float _baseMass;
    private float _pushTimer;
    private Collider[] _colliders;
    private PhysicMaterial _frictionMat;

    // --- 观测噪音 ---
    [Header("Noise")]
    public bool addNoise = true;
    public float gravityNoise = 0.05f;
    public float angularVelNoise = 0.05f;
    public float jointPosNoise = 0.01f;
    public float jointVelNoise = 0.075f;

    // --- 奖励权重 (对照 Python Config) ---
    const float w_tracking_lin_vel = 1.0f;
    const float w_tracking_ang_vel = 0.5f;
    const float w_lin_vel_z = -2.0f;
    const float w_ang_vel_xy = -0.05f;
    const float w_orientation = -1.0f;
    const float w_base_height = -10.0f;
    const float w_action_rate = -0.01f;
    const float w_alive = 0.15f;
    const float w_hip_pos = -1.0f;
    const float w_contact_no_vel = -0.2f;
    const float w_feet_swing_height = -20.0f;
    const float w_contact = 0.18f;

    public override void Initialize()
    {
        arts = GetComponentsInChildren<ArticulationBody>();
        
        LoadRobotConfig();
        
        // 解析关节
        legNum = 0; armNum = 0; waistNum = 0; headNum = 0;
        
        for (int k = 0; k < arts.Length; k++)
        {
            if (arts[k].jointType != ArticulationJointType.RevoluteJoint) continue;
            string jointname = arts[k].gameObject.name.ToLower();

            if (MatchKeywords(jointname, robotCfg.leg_keywords))
            {
                legs[legNum] = arts[k];
                legNum++;
            }
            else if (MatchKeywords(jointname, robotCfg.waist_keywords))
            {
                waists[waistNum] = arts[k];
                waistNum++;
            }
            else if (MatchKeywords(jointname, robotCfg.arm_keywords))
            {
                arms[armNum] = arts[k];
                armNum++;
            }
            else if (MatchKeywords(jointname, robotCfg.head_keywords))
            {
                heads[headNum] = arts[k];
                headNum++;
            }
        }

        body = arts[0].transform;
        _baseMass = arts[0].mass;
        _colliders = GetComponentsInChildren<Collider>();

        ApplyConfigGains();
        
        // 初始化数组大小
        System.Array.Resize(ref targetDeg, legNum);
        System.Array.Resize(ref defaultDeg, legNum);
        System.Array.Resize(ref lastActions, legNum);

        // 记录 Leg 的默认角度
        for (int i = 0; i < legNum; i++)
        {
             float defAngle = (robotCfg.leg_default_angles_rad != null && i < robotCfg.leg_default_angles_rad.Length)
                ? robotCfg.leg_default_angles_rad[i] * Mathf.Rad2Deg : 0f;
             defaultDeg[i] = defAngle;
        }
    }

    public override void OnEpisodeBegin()
    {
        episodeTime = 0f;
        _pushTimer = 0f;
        System.Array.Clear(lastActions, 0, lastActions.Length);

        // 随机重置
        if (train && !fixbody)
        {
            Vector3 startPos = new Vector3(Random.Range(-1f, 1f), baseHeightTarget + 0.1f, Random.Range(-1f, 1f));
            Quaternion startRot = Quaternion.Euler(0, Random.Range(-180f, 180f), 0);
            
            arts[0].TeleportRoot(startPos, startRot);
            arts[0].velocity = Vector3.zero;
            arts[0].angularVelocity = Vector3.zero;

            // 简单的将所有腿部关节复位
            for(int i=0; i<legNum; i++)
            {
                float angle = defaultDeg[i];
                ArticulationDrive drive = legs[i].xDrive;
                drive.target = angle;
                legs[i].xDrive = drive;
                legs[i].jointPosition = new ArticulationReducedSpace(angle * Mathf.Deg2Rad);
                legs[i].jointVelocity = new ArticulationReducedSpace(0f);
            }
        }

        // 随机指令
        if (train)
        {
            cmdLinVel = new Vector2(Random.Range(-0.8f, 0.8f), Random.Range(-0.4f, 0.4f)); // X前后, Y左右
            cmdAngVel = Random.Range(-0.8f, 0.8f);
        }
        else
        {
            cmdLinVel = Vector2.zero;
            cmdAngVel = 0f;
        }

        if (randomizeFriction) RandomizeFriction();
        if (randomizeBaseMass) RandomizeBaseMass();
    }

    // --- Actor 观测 (47维) ---
    public override void CollectObservations(VectorSensor sensor)
    {
        // 1. 机身角速度 (3)
        Vector3 localAngVel = body.InverseTransformDirection(arts[0].angularVelocity);
        sensor.AddObservation(ApplyVectorNoise(localAngVel, angularVelNoise));

        // 2. 重力投影 (3)
        Vector3 gravityVec = body.InverseTransformDirection(Vector3.down); 
        sensor.AddObservation(ApplyVectorNoise(gravityVec, gravityNoise));

        // 3. 目标指令 (3)
        sensor.AddObservation(cmdLinVel.x * 2.0f);
        sensor.AddObservation(cmdLinVel.y * 2.0f);
        sensor.AddObservation(cmdAngVel * 0.25f);

        // 4. 关节误差 + 速度 (12 + 12 = 24)
        for (int i = 0; i < legNum; i++)
        {
            float currentPosRad = legs[i].jointPosition[0]; 
            float defaultPosRad = defaultDeg[i] * Mathf.Deg2Rad;
            
            sensor.AddObservation(ApplyNoise(currentPosRad - defaultPosRad, jointPosNoise));
            sensor.AddObservation(ApplyNoise(legs[i].jointVelocity[0], jointVelNoise));
        }

        // 5. 上一帧动作 (12)
        for(int i=0; i<legNum; i++) sensor.AddObservation(lastActions[i]);

        // 6. 相位 (2)
        sensor.AddObservation(Mathf.Sin(2 * Mathf.PI * phase));
        sensor.AddObservation(Mathf.Cos(2 * Mathf.PI * phase));
    }

    public override void OnActionReceived(ActionBuffers actionBuffers)
    {
        var actions = actionBuffers.ContinuousActions;

        // 执行动作
        if (!fixbody)
        {
            for (int i = 0; i < legNum; i++)
            {
                targetDeg[i] = actions[i] * action_scale + defaultDeg[i];
                SetJointTargetDeg(legs[i], targetDeg[i], i);
            }
        }

        // 计算奖励
        CalculateRewards(actions);
        
        // 更新历史
        for(int i=0; i<legNum; i++) lastActions[i] = actions[i];
    }

    void CalculateRewards(ActionSegment<float> actions)
    {
        Vector3 localVel = body.InverseTransformDirection(arts[0].velocity);
        Vector3 localAngVel = body.InverseTransformDirection(arts[0].angularVelocity);
        
        // 1. Alive
        AddReward(w_alive);

        // 2. Tracking Linear Velocity
        float linVelError = Mathf.Pow(localVel.z - cmdLinVel.x, 2) + Mathf.Pow(localVel.x - cmdLinVel.y, 2);
        AddReward(w_tracking_lin_vel * Mathf.Exp(-linVelError / 0.25f));

        // 3. Tracking Angular Velocity
        float angVelError = Mathf.Pow(localAngVel.y - cmdAngVel, 2);
        AddReward(w_tracking_ang_vel * Mathf.Exp(-angVelError / 0.25f));

        // 4. Penalize Vertical Vel
        AddReward(w_lin_vel_z * Mathf.Pow(localVel.y, 2));

        // 5. Penalize Ang Vel XY
        AddReward(w_ang_vel_xy * (Mathf.Pow(localAngVel.x, 2) + Mathf.Pow(localAngVel.z, 2)));

        // 6. Orientation
        Vector3 gravityProj = body.InverseTransformDirection(Vector3.down);
        AddReward(w_orientation * (Mathf.Pow(gravityProj.x, 2) + Mathf.Pow(gravityProj.z, 2)));

        // 7. Base Height
        AddReward(w_base_height * Mathf.Pow(body.position.y - baseHeightTarget, 2));

        // 8. Action Rate
        float actionRateSum = 0f;
        for(int i=0; i<legNum; i++) actionRateSum += Mathf.Pow(actions[i] - lastActions[i], 2);
        AddReward(w_action_rate * actionRateSum);

        // 9. Hip Position (防止八字腿, 简单惩罚 Hip Roll/Yaw 偏离)
        // 注意：这里的索引 1,2,7,8 对应你 JSON 中的关节顺序，需根据实际情况调整
        // 假设 leg[1] 是 Left Hip Roll
        /* 
        if(legNum > 8) {
             AddReward(w_hip_pos * (Mathf.Pow(legs[1].jointPosition[0], 2) + Mathf.Pow(legs[7].jointPosition[0], 2)));
        }
        */

        // 10. Gait Rewards
        bool leftStance = phaseLeft < 0.55f;
        bool rightStance = phaseRight < 0.55f;

        // Contact
        if (leftFootSensor != null && leftStance == leftFootSensor.isGrounded) AddReward(w_contact);
        if (rightFootSensor != null && rightStance == rightFootSensor.isGrounded) AddReward(w_contact);

        // Swing Height
        if (leftFootTransform != null && !leftStance) 
            AddReward(w_feet_swing_height * Mathf.Pow(leftFootTransform.position.y - 0.08f, 2));
        if (rightFootTransform != null && !rightStance) 
            AddReward(w_feet_swing_height * Mathf.Pow(rightFootTransform.position.y - 0.08f, 2));

        // Contact No Vel
        if (leftFootSensor != null && leftFootSensor.isGrounded) 
            AddReward(w_contact_no_vel * leftFootSensor.contactVelocity.sqrMagnitude);
        if (rightFootSensor != null && rightFootSensor.isGrounded) 
            AddReward(w_contact_no_vel * rightFootSensor.contactVelocity.sqrMagnitude);
    }

    void FixedUpdate()
    {
        episodeTime += Time.fixedDeltaTime;
        phase = (episodeTime % stepPeriod) / stepPeriod;
        phaseLeft = phase;
        phaseRight = (phase + 0.5f) % 1.0f;

        if (pushRobots) HandleRandomPush();

        // 结束条件
        float pitch = Mathf.Abs(EulerTrans(body.eulerAngles[0]));
        float roll = Mathf.Abs(EulerTrans(body.eulerAngles[2]));
        
        if (pitch > 45f || roll > 45f || body.position.y < 0.35f)
        {
            EndEpisode();
        }
    }

    // --- 辅助功能 ---
    void LoadRobotConfig()
    {
        string fullPath = configPath;
        if (!File.Exists(fullPath)) fullPath = Path.Combine(Application.streamingAssetsPath, configPath);
        if (!File.Exists(fullPath)) fullPath = Path.Combine(Application.dataPath, "..", configPath); // 项目根目录

        if (File.Exists(fullPath))
        {
            robotCfg = JsonUtility.FromJson<RobotConfig>(File.ReadAllText(fullPath));
        }
        else
        {
            Debug.LogWarning("Config not found, using default");
            robotCfg = new RobotConfig();
        }
    }

    void ApplyConfigGains()
    {
        action_scale = robotCfg.action_scale;
        
        // 头部
        for (int i = 0; i < headNum; i++)
        {
            float defAngle = (robotCfg.head_default_angles_rad != null && i < robotCfg.head_default_angles_rad.Length) ? robotCfg.head_default_angles_rad[i] * Mathf.Rad2Deg : 0f;
            float kp = (robotCfg.head_kps != null && i < robotCfg.head_kps.Length) ? robotCfg.head_kps[i] : 100f;
            float kd = (robotCfg.head_kds != null && i < robotCfg.head_kds.Length) ? robotCfg.head_kds[i] : 2f;
            SetJointDriveWithParams(heads[i], defAngle, kp, kd);
        }
        // 手臂
        for (int i = 0; i < armNum; i++)
        {
            float defAngle = (robotCfg.arm_default_angles_rad != null && i < robotCfg.arm_default_angles_rad.Length) ? robotCfg.arm_default_angles_rad[i] * Mathf.Rad2Deg : 0f;
            float kp = (robotCfg.arm_kps != null && i < robotCfg.arm_kps.Length) ? robotCfg.arm_kps[i] : 100f;
            float kd = (robotCfg.arm_kds != null && i < robotCfg.arm_kds.Length) ? robotCfg.arm_kds[i] : 2f;
            SetJointDriveWithParams(arms[i], defAngle, kp, kd);
        }
        // 腰部
        for (int i = 0; i < waistNum; i++)
        {
            float defAngle = (robotCfg.waist_default_angles_rad != null && i < robotCfg.waist_default_angles_rad.Length) ? robotCfg.waist_default_angles_rad[i] * Mathf.Rad2Deg : 0f;
            float kp = (robotCfg.waist_kps != null && i < robotCfg.waist_kps.Length) ? robotCfg.waist_kps[i] : 100f;
            float kd = (robotCfg.waist_kds != null && i < robotCfg.waist_kds.Length) ? robotCfg.waist_kds[i] : 2f;
            SetJointDriveWithParams(waists[i], defAngle, kp, kd);
        }
        // 腿部 (PD会在 SetJointTargetDeg 中动态更新，这里只设初始值)
        for (int i = 0; i < legNum; i++)
        {
            float defAngle = (robotCfg.leg_default_angles_rad != null && i < robotCfg.leg_default_angles_rad.Length) ? robotCfg.leg_default_angles_rad[i] * Mathf.Rad2Deg : 0f;
            SetJointTargetDeg(legs[i], defAngle, i);
        }
    }

    void SetJointDriveWithParams(ArticulationBody joint, float target, float kp, float kd)
    {
        var drive = joint.xDrive;
        drive.stiffness = kp;
        drive.damping = kd;
        drive.target = target;
        joint.xDrive = drive;
    }

    void SetJointTargetDeg(ArticulationBody joint, float target, int idx)
    {
        var drive = joint.xDrive;
        if(robotCfg.leg_kps != null && idx < robotCfg.leg_kps.Length) drive.stiffness = robotCfg.leg_kps[idx];
        if(robotCfg.leg_kds != null && idx < robotCfg.leg_kds.Length) drive.damping = robotCfg.leg_kds[idx];
        drive.target = target;
        joint.xDrive = drive;
    }

    bool MatchKeywords(string name, string[] keywords)
    {
        if(keywords == null) return false;
        foreach(var k in keywords) if(name.Contains(k.ToLower())) return true;
        return false;
    }

    float EulerTrans(float angle)
    {
        angle %= 360f;
        if (angle > 180f) angle -= 360f;
        else if (angle < -180f) angle += 360f;
        return angle;
    }

    Vector3 ApplyVectorNoise(Vector3 value, float magnitude)
    {
        if (!addNoise) return value;
        return new Vector3(ApplyNoise(value.x, magnitude), ApplyNoise(value.y, magnitude), ApplyNoise(value.z, magnitude));
    }

    float ApplyNoise(float val, float mag) => addNoise ? val + Random.Range(-mag, mag) : val;

    void RandomizeFriction()
    {
        if (_colliders == null) return;
        if (_frictionMat == null)
        {
             _frictionMat = new PhysicMaterial { name = "RandFriction", frictionCombine = PhysicMaterialCombine.Multiply };
             foreach(var c in _colliders) c.material = _frictionMat;
        }
        float f = Random.Range(frictionRange.x, frictionRange.y);
        _frictionMat.staticFriction = f; 
        _frictionMat.dynamicFriction = f;
    }

    void RandomizeBaseMass()
    {
        if(arts != null && arts.Length > 0)
            arts[0].mass = Mathf.Max(1.0f, _baseMass + Random.Range(baseMassDeltaRange.x, baseMassDeltaRange.y));
    }

    void HandleRandomPush()
    {
        _pushTimer += Time.fixedDeltaTime;
        if (_pushTimer >= pushIntervalSeconds)
        {
            _pushTimer = 0f;
            Vector2 dir = Random.insideUnitCircle.normalized;
            arts[0].AddForce(new Vector3(dir.x, 0, dir.y) * Random.Range(0.5f, maxPushVelocity), ForceMode.VelocityChange);
        }
    }
}