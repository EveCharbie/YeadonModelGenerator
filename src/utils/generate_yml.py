def generate_yml(pelvis: float, knuckle: float, pike_hand: float, tuck_hand: float): #TODO add markers for the tuck position
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
      translations: xyz
      rotations: xyz
      markers:
        HeadTop:
          position: [0, 0, {pelvis}] # change depending on the length of hip_joint to top_of_the_head
        PelvisBase:
          position: [0, 0, 0]

    Stomach:
      meshfile: Model_mesh/boule.stl
      meshrt: [-0.15, 0, 0]
      meshxyz: [0, -0.025, 0]
      rotations: xyz

    Rib:
      meshfile: Model_mesh/thorax.stl
      meshrt: [-0.15, 0, 0]
      meshxyz: [0, -0.025, 0]
      rotations: xyz
      
    Nipple:
      meshfile: Model_mesh/boule.stl
      meshrt: [-0.15, 0, 0]
      meshxyz: [0, -0.025, 0]
      rotations: xyz
      
    Shoulder:
      meshfile: Model_mesh/boule.stl
      meshrt: [-0.15, 0, 0]
      meshxyz: [0, -0.025, 0]
      rotations: xyz

    Head:
      meshfile: Model_mesh/tete.stl
      meshrt: [0, 0, pi]
      meshxyz: [0, 0, 0]
      rotations: xyz

    Eyes:
      meshfile: Model_mesh/boule.stl
      meshrt: [0, 0, 0]
      meshxyz: [0, 0, 0]
      rotations: xyz
    
    RightUpperArm:
      meshfile: Model_mesh/bras.stl
      rotations: xyz
      markers:
        RightShoulder:
          position: [0, 0, 0]
    
    RightForearm:
      meshfile: Model_mesh/avantbras.stl
      rotations: xyz
      markers:
        RightElbow:
          position: [0, 0, 0]
    
    RightHand:
      meshfile: Model_mesh/main.stl
      rotations: xyz
      markers:
        MiddleRightHand:
          position: [0, 0, -0.1]
        RightKnuckle:
          position: [0, 0, {knuckle}] # length wrist-knuckle
    
    LeftUpperArm:
      meshfile: Model_mesh/bras.stl
      rotations: xyz
      markers:
        LeftShoulder:
          position: [0, 0, 0]
    
    LeftForearm:
      meshfile: Model_mesh/avantbras.stl
      rotations: xyz
      markers:
        LeftElbow:
          position: [0, 0, 0]
    
    LeftHand:
      meshfile: Model_mesh/main.stl
      rotations: xyz
      markers:
        MiddleLeftHand:
          position: [0, 0, -0.1]
        LeftKnuckle:
          position: [0, 0, {knuckle}] # length wrist-knuckle
    
    UpperLegs:
      meshfile: Model_mesh/cuisse.stl
      rotations: xyz
    
    LowerLegs:
      meshfile: Model_mesh/jambe.stl
      meshrt: [pi, 0, 0]
      meshxyz: [0, 0, 0]
      rotations: xyz
      markers: 
        PikeTargetRightHand:
          position: [-0.1, 0, {pike_hand}] # pike knee to hand
        PikeTargetLeftHand:
          position: [0.1, 0, {pike_hand}]
      markers: 
        TuckTargetRightHand:
          position: [-0.1, 0, {tuck_hand}] # tuck knee to hand
        TuckTargetLeftHand:
          position: [0.1, 0, {tuck_hand}]
        Knee:
          position: [0, 0, 0]
    
    Feet:
      rt: [-0.35, 0, 0]
      meshfile: Model_mesh/pied.stl
      meshrt: [0, 0, pi]
      meshxyz: [0, 0, 0]
      rotations: xyz
      markers:
        Ankle:
          position: [0, 0, 0]
    """
    with open('src/biomake/tech_opt.yml', 'w') as file:
        file.write(content)