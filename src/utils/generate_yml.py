def generate_yml(pelvis: float, knuckle: float, knee: float):
    content = f"""
    # bioMod configuration for models used with TechOpt83
    #
    # This file is used by biomake to generate the bioMod from measurements.
    #
    Human:
      fused: True
      gravity: [0, 0, -9.81]
    
    Pelvis:
      meshfile: Model_mesh/pelvis.stl
      meshrt: [-0.175, 0, 0]
      meshxyz: [0, 0, 0]
      markers:
        HeadTop:
          position: [0, 0, {pelvis}] # change depending on the length of hip_joint to top_of_the_head
        PelvisBase:
          position: [0, 0, 0]
    
    Thorax:
      meshfile: Model_mesh/thorax.stl
      meshrt: [-0.15, 0, 0]
      meshxyz: [0, -0.025, 0]
    
    Head:
      meshfile: Model_mesh/tete.stl
      meshrt: [0, 0, pi]
      meshxyz: [0, 0, 0]
    
    RightUpperArm:
      meshfile: Model_mesh/bras.stl
      markers:
        RightShoulder:
          position: [0, 0, 0]
    
    RightForearm:
      meshfile: Model_mesh/avantbras.stl
      markers:
        RightElbow:
          position: [0, 0, 0]
    
    RightHand:
      meshfile: Model_mesh/main.stl
      markers:
        MiddleRightHand:
          position: [0, 0, -0.1]
        RightKnuckle:
          position: [0, 0, {knuckle}] # length wrist-knuckle
    
    LeftUpperArm:
      meshfile: Model_mesh/bras.stl
      markers:
        LeftShoulder:
          position: [0, 0, 0]
    
    LeftForearm:
      meshfile: Model_mesh/avantbras.stl
      markers:
        LeftElbow:
          position: [0, 0, 0]
    
    LeftHand:
      meshfile: Model_mesh/main.stl
      markers:
        MiddleLeftHand:
          position: [0, 0, -0.1]
        LeftKnuckle:
          position: [0, 0, {knuckle}] # length wrist-knuckle
    
    UpperLegs:
      meshfile: Model_mesh/cuisse.stl
    
    LowerLegs:
      meshfile: Model_mesh/jambe.stl
      meshrt: [pi, 0, 0]
      meshxyz: [0, 0, 0]
      markers:
        TargetRightHand:
          position: [-0.1, 0, {knee}] # pike knee to hand
        TargetLeftHand:
          position: [0.1, 0, {knee}]
        Knee:
          position: [0, 0, 0]
    
    Feet:
      rt: [-0.35, 0, 0]
      meshfile: Model_mesh/pied.stl
      meshrt: [0, 0, pi]
      meshxyz: [0, 0, 0]
      markers:
        Ankle:
          position: [0, 0, 0]
    """
    with open('tech_opt.yml', 'w') as file:
        file.write(content)