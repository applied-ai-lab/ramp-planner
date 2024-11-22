"""
    Copyright (C) 2023 The Manufacturing Technology Centre
    Author: Mark Robson

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple
import networkx as nx
import logging
import xml.etree.ElementTree as ET
import numpy as np
import trimesh
from matplotlib import pyplot as plt

from sparc_planning.src.al_structures import BasicSort, Sort

class ElementType(Enum):
    LINK = 0
    IN_M_END = 1
    IN_F_END = 2
    THRU_M = 3
    ANGLE_F = 4
    IN_F = 5
    THRU_F = 6
    IN_M_END_FEET = 7
    ANGLE_M_END = 8

    @staticmethod
    def as_dict():
        return {
            "link": 0,
            "in-m-end": 1,
            "in-f-end": 2,
            "thru-m": 3,
            "angle-f": 4,
            "in-f": 5,
            "thru-f": 6,
            "in-m-end-feet": 7,
            "angle-m-end": 8,
            }

# offsets for stacking up with other parts
OFFSET_DATA = {
    ElementType.THRU_M: 17.6,
    ElementType.IN_M_END: 8.6,
    ElementType.ANGLE_F: 33.0,
    ElementType.IN_F: 17.6,
    ElementType.THRU_F: 17.6,
    ElementType.IN_F_END: 23.200,
    ElementType.LINK:0.0,
    ElementType.IN_M_END_FEET: 8.6,
    ElementType.ANGLE_M_END: 8.6,
}

# offsets for specifying hole locations in parts (distance in z from object centre) - change to 
PEG_OFFSET_DATA = {
    ElementType.THRU_M: 0.0,
    ElementType.IN_M_END: 0.0,
    ElementType.ANGLE_F: 0.0,
    ElementType.IN_F: 0.0,
    ElementType.THRU_F: 0.0,
    ElementType.IN_F_END: 0.0,
    ElementType.IN_M_END_FEET: 0.0,
    ElementType.ANGLE_M_END: 0.0,
}

# offsets in [x,y,z] to top right corner of the object tag from object center
TAG_POSITION_DATA = { #TODO
    ElementType.IN_M_END:[12.750, -29.600, 26.600], 
    ElementType.IN_F_END:[12.750,-17.150,29.900],
    ElementType.IN_M_END_FEET: [12.750, -29.600, 26.600],
    ElementType.ANGLE_M_END:[12.750, -29.600, 26.600], 
}

MODEL_DATA = {
    ElementType.THRU_M: "models/thru-m.stl",
    ElementType.IN_M_END: "models/in-m-end.stl",
    ElementType.IN_M_END_FEET: "models/in-m-end-feet.stl",
    ElementType.ANGLE_F: "models/angle-f.stl",
    ElementType.IN_F: "models/in-f.stl",
    ElementType.THRU_F: "models/thru.stl",
    ElementType.IN_F_END: "models/in-f-end.stl",
    ElementType.LINK: "models/20x20x200.stl",
    ElementType.ANGLE_M_END: "models/in-m-end.stl",
}

def element_type_from_string(name:str) -> ElementType:
    d = ElementType.as_dict()
    if name in d: return ElementType(d[name])
    raise ValueError(f'{name} is not a valid element type string\nvalid strings: {d.keys()}')

@dataclass
class BeamComponent:
    name:str
    type:ElementType
    child:Optional[BeamComponent]
    parent:Optional[BeamComponent]
    length:Optional[float]
    marker:Optional[int]

    def is_middle(self) -> bool:
        if self.type.value in [3,4,5,6]:
            return True
        return False

    def is_male(self) -> bool: 
        if self.type.value in [1,7,3,8]:
            return True
        return False
    
    def is_link(self) -> bool:
        if self.type.value == 0:
            return True
        return False
    
    def is_angle(self) -> bool:
        if self.type.value in [8,4]:
            return True
        return False

    def check_valid(self):
        # check Connection validity
        if self.is_link():
            if isinstance(self.parent, type(None)) or isinstance(self.child,type(None)):
                raise ValueError("Components with element type LINK must have both a parent and child joint")
            if self.child.is_link():
                raise ValueError("Components with element type LINK cannot have child type LINK.")
            if self.parent.is_link():
                raise ValueError("Components with element type LINK cannot have child type LINK.")
            if isinstance(self.length, type(None)) or (self.length < 0):
                raise ValueError("Components with element type LINK must have +ve length specified")
        else:
            if not isinstance(self.child,type(None)): # if joint has a child link
                if not self.child.is_link(): # make sure it is a link
                    raise ValueError("Components with element type JOINT must have child type LINK.")
            if not isinstance(self.parent,type(None)): # if joint has a parent link
                if not self.parent.is_link(): # make sure it is a link
                    raise ValueError("Components with element type JOINT must have parent type LINK.")
            if isinstance(self.child,type(None)) and isinstance(self.parent,type(None)):
                raise ValueError("joints must be attached to a link")
            if not isinstance(self.length,type(None)):
                logging.INFO(f"length ({self.length}) will be ignored as JOINT elements have set lengths")
        return True

    def load_model(self) -> trimesh.Trimesh:
        if self.is_link():
            # m = trimesh.creation.box((0.02,0.02,self.length/1000.0)) # units in meters
            scale_mat = np.eye(4)
            scale_mat[2, 2] = self.length / 100.0  # scale z direction to self.length in meters
            m = trimesh.load_mesh(MODEL_DATA[self.type]).apply_transform(scale_mat)
            t = np.eye(4)
            t[2,3]=-m.bounds[1,2]/2.0
            m.apply_transform(t)
        else:
            #scale to meters
            scale_mat = np.eye(4)#*0.001
            m = trimesh.load_mesh(MODEL_DATA[self.type]).apply_transform(scale_mat)
            # adjust origin to centre of object bounds
            t = np.eye(4)
            # if (self.type == ElementType.LINK):
            t[2,3]=-m.bounds[1,2]/2.0
            m.apply_transform(t)
            # some designs are in a different orientation to the others - we want y upwards for simulator world
            # m.apply_transform(np.asarray([[0,0,1,0],[0,1,0,0],[-1,0,0,0],[0,0,0,1]]))
            m.apply_transform(np.asarray([[0,-1,0,0],[-1,0,0,0],[0,0,1,0],[0,0,0,0]])) #rotate 90 degrees about z
            m.apply_transform(np.asarray([[0,0,-1,0],[0,1,0,0],[1,0,0,0],[0,0,0,0]])) #rotate 90 degrees about y
            m.apply_transform(np.asarray([[0,-1,0,0],[-1,0,0,0],[0,0,1,0],[0,0,0,0]])) #rotate 90 degrees about z
            if (self.type == ElementType.IN_F_END):
                m.apply_transform(np.asarray([[-1,0,0,0],[0,-1,0,0],[0,0,1,0],[0,0,0,0]]))# rotate 180 degrees about x
                # m.apply_transform(np.asarray([[-1,0,0,0],[0,-1,0,0],[0,0,1,0],[0,0,0,1]]))
            # else:
            #     # rotate 90 degrees about z
            #     m.apply_transform(np.asarray([[0,-1,0,0],[-1,0,0,0],[0,0,1,0],[0,0,0,1]]))
        #print(m.bounds)
        return m

    def __repr__(self):
        ret =  f'name={self.name}, type={self.type},'
        if not isinstance(self.parent, type(None)):
            ret += f' parent={self.parent.name},'
        if not isinstance(self.child, type(None)):
            ret += f' child={self.child.name},'
        if not isinstance(self.length, type(None)):
            ret += f' length={self.length},'
        if not isinstance(self.marker, type(None)):
            ret += f' marker={self.marker},'
        return ret


@dataclass
class Beam:
    name:str
    joints:list[BeamComponent]
    links:list[BeamComponent]
    origin:np.ndarray = np.eye(4)
    flipped:bool=False

    def __post_init__(self):
        # check list validity
        for joint in self.joints:
            if joint.is_link():
                raise ValueError("One or more joints is of ElementType LINK.")
        for link in self.links:
            if not link.is_link():
                raise ValueError("One or more links is NOT of ElementType LINK.")

    def get_structure(self) -> list[BeamComponent]:
        """gets ordered list of beam components"""
        if len(self.joints) == 0:
            logging.ERROR('Beam has no joint components!')
            return []
        n = [self.joints[0]] # add first joint to list (not necessarily in order yet)
        complete = False
        while not complete:
            if not isinstance(n[0].parent, type(None)): # if the part has a parent
                n.insert(0, n[0].parent) # add to start of list
            elif not isinstance(n[-1].child,type(None)): # if it has not patent but has a child
                n.append(n[-1].child) # add to end of list
            else: # no more nodes to add to list
                complete = True
        return n

    def get_component_type_list(self) -> list[int]:
        return [e.type.value for e in self.get_structure()]
    
    def get_num_components(self) -> int:
        return len(self.get_structure())

    def get_component_by_name(self, name:str) ->BeamComponent:
        """finds component by name else raises a ValueError if component not in beam"""
        l = self.get_structure()
        for comp in l:
            if comp.name == name: return comp
        raise ValueError(f"no component {name} in {self.name}")

    def get_component_offsets(self) -> list[np.ndarray]:
        """returns offset pose data for each beam component as affine matrix
        """
        components = self.get_structure()
        poses = []
        current_pose = np.eye(4)
        for c in components:
            offset = OFFSET_DATA[c.type]
            current_pose[2,3] += offset/1000.0
            if c.is_link():
                current_pose[2,3] += c.length/2000.0
            poses.append(np.copy(current_pose))
            if c.is_link():
                current_pose[2,3] += c.length/2000.0
            if c.is_middle():
                current_pose[2,3] += offset/1000.0
        if components[-1].type == ElementType.IN_F_END:
            poses[-1][:3,:3] = [[1,0,0],[0,-1,0],[0,0,-1]] # rotate last end piece 180deg about x
        else:
            poses[-1][:3,:3] = [[-1,0,0],[0,1,0],[0,0,-1]]#rotate last end piece 180 about y
        poses[0]=np.eye(4) # zero first pose
        return poses
    
    def create_display_scene(self, colorise:bool=False):
        """ create Trimesh scene containing the meshes of parts arranged in the beam configuration
            optionally, cycle through RGB colors to make parts visually distinct
        """
        pos = self.get_component_offsets()
        comps = self.get_structure()
        scene = trimesh.Scene()
        cols = [[1,0,0],[0,1,0],[0,0,1]]
        i = 0
        for p,c in zip(pos,comps):
            mesh = c.load_model() #load mesh
            if colorise:
                mesh_colors = np.ones((len(mesh.faces), 4)) #create array for face color values
                mesh_colors[:,:3]=cols[i]
                i+=1
                if i>=len(cols): i=0
                mesh.visual.face_colors = mesh_colors # apply color
            scene.add_geometry(mesh, transform=p)
            #print(c.name, p)
        return scene

    def __repr__(self):
        return f"Beam: '{self.name}' --> {self.get_component_type_list()}"

    def get_sparc_representation(self) -> Tuple[str,dict,List[str],List[str]]:
        """extract beam data for sparc program (needs to be put into correct places!)
        Returns:
            str: beam_name (add to list of sort:beam)
            dict: beam_components sorted by type)
            str: connections (statics, just add to program)
            str: component relations (statics, just add to program)
        """
        parts_in_order = [g.name for g in self.get_structure()]
        t = self.get_component_type_list()
        d = ElementType.as_dict()
        jd = {k: [] for k in d.keys()}
        for i in range(len(t)):
            key = [*d.keys()][t[i]]
            jd[key].append(parts_in_order[i])
        component_relations = [f'component({self.name},{comp}).' for comp in parts_in_order]
        connection_relations = []
        for j in range(len(t)-1):
            connection_relations.append(f"connected_to({parts_in_order[j]},{parts_in_order[j+1]}).")
        return self.name, jd, connection_relations, component_relations
    
    def get_connection_offsets(self)->dict[str:np.ndarray]:
        """hole offset positions by joint name for all joints in beam
        Returns:
            dict[str:np.ndarray]: {joint_name:SE3, ...}
        """
        pos = self.get_component_offsets()
        comps = self.get_structure()
        joints = [(j, comps[j]) for j in range(len(comps)) if not comps[j].is_link()] # find joint components
        names = [j.name for (_,j) in joints]
        hole_positions = []
        for (i,j) in joints:
            offset = np.eye(4)
            offset[2,3] += PEG_OFFSET_DATA[j.type]/1000.0 # convert to meters
            hole_positions.append(np.matmul(pos[i],offset)) # use transform multiplication to allow for rotation changes
        return dict(zip(names,hole_positions))
    
    def translate(self, offset) -> None:
        """translate beam using x,y,z list or np.array"""
        assert len(offset) == 3, "Expected offset length == 3: for x,y and z directions"
        self.origin[:3,3] += offset
    
    def rotate(self, rot_mat:np.ndarray, centre:list[float]=[0,0,0]) -> None:
        '''perform 3d rotation of beam using rotation matrix'''
        assert rot_mat.shape == (3,3), "Rotation matrix must be 3x3 array"
        assert len(centre) == 3, "Centre must be x,y,z coordinates"
        o = np.eye(4)
        o[:3,:3]=rot_mat
        o[:3,3]=centre
        self.origin = np.matmul(o, self.origin)

    def transform(self, affine:np.ndarray) -> None:
        '''affine transform using 4x4 array'''
        assert affine.shape == (4,4), "requires 4x4 array input"
        self.origin = np.matmul(self.origin, affine)

@dataclass
class Connection:
    element1:Beam
    joint1:BeamComponent
    element2:Beam
    joint2:BeamComponent

    def __post_init__(self):
        # check valid connection
        if self.joint1.is_link() or self.joint2.is_link():
            raise ValueError("Beams can only be connected at joints")
        if self.element1.name == self.element2.name:
            raise ValueError("Beams cannot be connected to themselves")

        if self.joint1.is_male():
            if self.joint2.is_male():
                raise ValueError(f'These beam components are not compatible (M:M)\n{self.joint1.name}/{self.joint2.name}')
        else:
            if not self.joint2.is_male():
                logging.debug(self.joint1.type, self.joint1.type.value , f'is_male:{self.joint1.is_male()}')
                logging.debug(self.joint2.type, self.joint2.type.value , f'is_male:{self.joint2.is_male()}')
                raise ValueError(f'These beam components are not compatible (F:F)\n{self.joint1.name}/{self.joint2.name}')

    def __repr__(self):
        return f"Connection: {self.element1.name}/{self.joint1.name} connects to {self.element2.name}/{self.joint2.name}\n"
    
    def to_sparc_coarse(self) -> str:
        if self.joint1.is_male():
            if self.joint1.type.value == 3:
                return(f'fits_through_c({self.element1.name},{self.element2.name}).')
            else:
                return(f'fits_into_c({self.element1.name},{self.element2.name}).')
        else:
            if self.joint2.type.value == 3:
                return(f'fits_through_c({self.element2.name},{self.element1.name}).')
            else:
                return(f'fits_into_c({self.element2.name},{self.element1.name}).')
            
    def to_sparc_fine(self) -> str:
        if self.joint1.is_male():
            if self.joint1.type.value == 3:
                return(f'fits_through_f({self.joint1.name},{self.joint2.name}).')
            else:
                return(f'fits_into_f({self.joint1.name},{self.joint2.name}).')
        else:
            if self.joint2.type.value == 3:
                return(f'fits_through_f({self.joint2.name},{self.joint1.name}).')
            else:
                return(f'fits_into_f({self.joint2.name},{self.joint1.name}).')
        
@dataclass
class BeamAssembly:
    connections:list[Connection]
    base:Beam

    def __post_init__(self):
        jlist = [con.joint1 for con in self.connections] + [con.joint2 for con in self.connections]
        duplicates = []
        for i in jlist:
                if jlist.count(i) > 1:
                    duplicates.append(i)
        if len(duplicates) > 0:
            raise ValueError(f'Some joints are connected twice! duplicates: {duplicates}')
        self.beams = self.get_beams()

    def get_graph(self):
        """create graphx graph of the assembly with beams as nodes and connections as edges"""
        logging.debug('Called BeamAssembly.get_graph()')
        if len(self.connections) == 0:
            logging.ERROR('No connections specified')
            return nx.MultiGraph() # empty graph
        # create graph
        edges = [(con.element1.name, con.element2.name) for con in self.connections]
        g = nx.MultiGraph(edges)
        # add connection data for each edge
        g_edges = [(con.element1.name, con.element2.name, 0) for con in self.connections]
        edge_data = [{"b1_joint":con.joint1.name, "b2_joint":con.joint2.name, "conn":con} for con in self.connections]
        attrs = dict(zip(g_edges,edge_data))
        logging.debug(attrs)
        nx.set_edge_attributes(g,attrs)
        return g
    
    def get_beams(self) -> list[Beam]:
        """extracts a list of beams in the assembly from the connection data"""
        beams = [self.base]
        for connection in self.connections:
            if connection.element1 not in beams:
                beams.append(connection.element1)
            if connection.element2 not in beams:
                beams.append(connection.element2)
        #update internal beam list
        self.beams = beams
        return beams

    def find_connection_between_beams(self, beam1:str, beam2:str) -> Connection|None:
        """returns connection between beams, or None if no connection exists
        Args:
            beam1 (str): 1st beam name
            beam2 (str): 2nd beam name
        Returns:
            Connection|None
        """
        G = self.get_graph()
        data = G.get_edge_data(beam1,beam2)
        if not isinstance(data,type(None)):
            return data[0]["conn"]         
        logging.error("No connection between these two beams was found in the assembly description")
        return None
    
    def get_beam_connections(self,beam_name:str) ->list[Connection]|None:
        """gets list of all connections to the named beam or None if beam not in assembly"""
        G = self.get_graph()
        edge_data = [G.get_edge_data(beam_name,b.name) for b in self.beams]
        mask = [not isinstance(a,type(None)) for a in edge_data]
        if np.any(mask) == False:
            logging.info(f"No connection found to beam: {beam_name}")
            return None
        return [x[0]['conn'] for x, y in zip(edge_data, mask) if y == True]
    
    def get_beam_by_name(self,name:str) -> Beam|None:
        """gets beam by name if in assembly, otherwise returns None"""
        for beam in self.beams:
            if beam.name == name:
                return beam
        logging.info(f"No beam named {name} found in this assembly")
        return None
    
    def get_beam_by_component_name(self, component_name:str) -> Beam|None:
        """finds beam by component part, returns None if component
        is not in any beam in the assembly """
        for beam in self.beams:
            try: 
                _ = beam.get_component_by_name(component_name)
                return beam
            except ValueError:
                continue
        return None

    def get_beam_rotation_state(self) -> dict[str:str]:
        """labels beams as 'horizontal', 'vertical', or 'angled'"""
        placed = [self.base.name]
        results = {self.base.name:'horizontal'}
        beam_names=[beam.name for beam in self.beams]
        for b in beam_names:
            conns = self.get_beam_connections(b)
            # loop through all connections to this beam
            for conn in conns:
                # figure out what the new beam is called
                if conn.element1.name == b:
                    connecting_beam = conn.element2.name
                else:
                    connecting_beam = conn.element1.name
                # check that this beam hasnt already been assigned
                if connecting_beam not in placed:
                    if b in results:
                        # assign to a bin if we can
                        if conn.joint1.is_angle() or conn.joint2.is_angle():
                            results[connecting_beam]='angled'
                            placed.append(connecting_beam)
                        elif results[b] == 'horizontal': 
                            results[connecting_beam]='vertical'
                            placed.append(connecting_beam)
                        elif results[b] == 'vertical': 
                            results[connecting_beam]='horizontal'
                            placed.append(connecting_beam)
        return results   

    def get_target_beam_positons(self) -> dict[str:np.ndarray]:
        """Returns a dictionary keyed by beam name with value of positions relative
        to the base beam (as 4x4 affine transform) also updates all beam.origin positions.
        Provides no guarantee on length connectness for joint positions on angled beams.
        """
        self.joint_dictionary = {}
        logging.info('Updating beam positions...')
        # reset all beams to origin pose by default
        for beam in self.beams:
            beam.origin = np.eye(4)
        bg = self.get_graph() # graph with beams as nodes and connections as edges
        rot_data = self.get_beam_rotation_state()
        # aim is to visit all nodes in graph, starting at the base beam, extract position of each beam
        current_beam = self.base.name
        visited_beams = []
        updated_beams = [self.base.name]
        not_yet_visited = []
        running = True
        while running: # loop search
            for conn in bg[current_beam]:
                #print(conn)
                if conn not in updated_beams:
                    c1 = bg[current_beam][conn][0]["b1_joint"] # beam 1 joint name where new beam in connected
                    c2 = bg[current_beam][conn][0]["b2_joint"] # beam 2 joint name which connects to c1
                    b1 = self.get_beam_by_name(current_beam)
                    # check that connections are associated to the right beam
                    if c1 not in [j.name for j in b1.joints]:
                        logging.info("switched components!")
                        c1,c2 = c2,c1
                    # calculate target position from connection offset and beam origin
                    logging.info(f"searching {b1.name} for component {c1}")
                    c1_offset = b1.get_connection_offsets()[c1]
                    q = np.matmul(b1.origin,c1_offset)
                    # get connecting beam (b2) data
                    logging.info(f'Repositioning {conn} ({c2})')
                    connecting_beam = self.get_beam_by_name(conn)
                    #connecting beam origin should be I by default
                    p = connecting_beam.get_connection_offsets()[c2]
                    logging.debug(f'initial connection position ({c2}):',p)
                    # roll 180 about z axis if specified
                    if connecting_beam.flipped:
                        connecting_beam.origin = np.asarray([[-1,0,0,0],[0,-1,0,0],[0,0,1,0],[0,0,0,1]],np.float32)
                    # update b2 position
                    # first translate connection hole position to origin
                    connecting_beam.translate(-p[:3,3])
                    # rotate about origin to neutral position
                    #connecting_beam.rotate(np.linalg.inv(p[:3,:3]))
                    # based on connection type set up horizontal/vertical rotations
                    if rot_data[conn] == 'vertical':
                        logging.debug('vertical')
                        theta = np.pi/2 # rotate 90 degrees
                    else:
                        theta = 0
                    # 'angled' assume 0 degrees for now (will need to fix later)
                    # rotate about y (up vector)
                    x = np.asarray([[np.cos(theta), 0, np.sin(theta)],[0,1,0],[-np.sin(theta), 0, np.cos(theta)]])
                    #q[:3,:3] = x
                    connecting_beam.rotate(x)
                    # translate to match new hole position
                    connecting_beam.translate(q[:3,3])
                    self.joint_dictionary[conn] = q[:3, 3]
                    logging.debug('target connection positon:', q[:3,3])
                    #connecting_beam.transform(np.matmul(np.linalg.inv(p),q))
                    # record that we already moved this beam
                    updated_beams.append(conn)
                    # add to list of possible future beams
                    if conn not in visited_beams: not_yet_visited.append(conn)
            # note we have completed this beam's connections
            visited_beams.append(current_beam)
            # if there are no more beams left to update, break loop
            if len(not_yet_visited)==0:
                running = False
            else:
                # otherwise get a new "current beam", from list of not_yet_visited beams
                current_beam = not_yet_visited.pop(0) # pop then removes it from the list
        
        # loop through beams
        for beam in self.beams:
            # find "angled beams"
            if rot_data[beam.name] == 'angled':
                logging.info(f'Rotating angled beam ({beam.name})...')
                # get end connections
                connections = self.get_beam_connections(beam.name)
                connection1 = connections[0]
                connection2 = connections[-1]
                # find end connection x,y positions (assume z1==z2)
                if connection1.element1.name == beam.name:
                    bm1 = connection1.element2
                    cn1 = connection1.joint2
                    cn0 = connection1.joint1
                else:
                    bm1 = connection1.element1
                    cn1 = connection1.joint1
                    cn0 = connection1.joint2
                if connection2.element1.name == beam.name:
                    bm2 = connection2.element2
                    cn2 = connection2.joint2
                else:
                    bm2 = connection2.element1
                    cn2 = connection2.joint1

                bm1_comp_pos = bm1.get_connection_offsets()[cn1.name]
                bm2_comp_pos = bm2.get_connection_offsets()[cn2.name]
                conn1 = np.matmul(bm1.origin, bm1_comp_pos)
                #print(conn1)
                conn2 = np.matmul(bm2.origin, bm2_comp_pos)
                #print(conn2)
                x1,y1 = conn1[0,3], conn1[2,3]
                x2,y2 = conn2[0,3], conn2[2,3]
                # calculate angle @connection 1 (assumes a right angle triangle)
                alpha = np.arctan2(x2-x1, y2-y1)
                # reposition beam to correct angle
                #translate joint to origin
                beam.origin = np.eye(4)
                # flip about x axis if specified
                if beam.flipped:
                    beam.origin = np.asarray([[-1,0,0,0],[0,-1,0,0],[0,0,1,0],[0,0,0,1]], np.float32)
                c0 = beam.get_connection_offsets()[cn0.name]
                beam.translate(-c0[:3,3])
                # rotate to new angle
                logging.info(f'Angle: {np.degrees(alpha):.02f} degrees')
                beam.rotate(np.asarray([[np.cos(alpha), 0, np.sin(alpha)],[0,1,0],[-np.sin(alpha), 0, np.cos(alpha)]]))
                #translate to new position
                beam.translate(conn1[:3,3])
                # NOTE this provides no guarantee on length connectness for other joint positions of angle beam
        
        return dict([(b.name, b.origin) for b in self.beams])

    def create_display_scene(self, colorise=False)->trimesh.Scene:
        """minimal example of a trimesh scene for displaying beam assembly"""
        # update beam positions
        d = self.get_target_beam_positons()
        logging.debug(d)
        cols = [[1,0,0],[0,1,0],[0,0,1],[1,1,0]]
        i = 0
        # create scene
        scene = trimesh.Scene()
        for beam in self.beams:
            comps = beam.get_structure()
            pos = beam.get_component_offsets()
            for p,c in zip(pos,comps):
                # load mesh
                mesh = c.load_model()
                # calculate component world pose
                t = np.matmul(beam.origin, p)
                if colorise:
                    mesh_colors = np.ones((len(mesh.faces), 4)) #create array for face color values
                    if i>=len(cols): i=0
                    mesh_colors[:,:3]=cols[i]
                    mesh.visual.face_colors = mesh_colors # apply color
                scene.add_geometry(mesh, transform=t)
            i+=1
        return scene
    

#!FUNCTIONS:
    
def load_assembly_xml(beams:list[Beam],assem_xml_file:str) -> BeamAssembly:
    """loads assembly from xml file

    Args:
        beams (list[Beam]): set of possible beams (must include all beams in assembly)
        assem_xml_file (str): file location

    Raises:
        ValueError: if beam named in assembly connections is not in beams
        ValueError: if no base beam specified
        ValueError: if component specified in assembly connection is not in beam

    Returns:
        BeamAssembly
    
    Xml example with 4 beams joined in a loop, beams are specified seperately
    to aid reuse across assemblies. 

    NOTE beams and connections must have unique names

    <assembly>
        <component beam="beam1" base="True"/>
        <component beam="beam2" />
        <component beam="beam3" flipped="True"/>
        <component beam="beam4" />
        <connection name="C1">
            <element component="beam1" joint="J1" />
            <element component="beam2" joint="J2" />
        </connection>
        <connection name="C2">
            <element component="beam1" joint="J3" />
            <element component="beam4" joint="J4" />
        </connection>
        <connection name="C3">
            <element component="beam2" joint="J5" />
            <element component="beam3" joint="J6" />
        </connection>
        <connection name="C4">
            <element component="beam3" joint="J7" />
            <element component="beam4" joint="J8" />
        </connection>
    </assembly>
    """
    logging.info(f"Processing assembly file: {assem_xml_file}")
    connections = []
    tree = ET.parse(assem_xml_file)
    root = tree.getroot()

    def find_beam_by_name(name:str, beams:list[Beam])->Beam:
        for beam in beams:
            if beam.name == name: return beam
        raise ValueError(f"No beam {name} in scope")
    
    # get beam details first
    base = None
    for component in root.findall('component'):
        #print(component.get('beam'), component.get('base'))
        b = find_beam_by_name(component.get('beam'), beams)
        # identify base beam
        if component.get('base'):
            base = b
        # set any flipped beams for this assembly
        if component.get('flipped'):
            b.flipped = True
        else:
            b.flipped = False

    logging.info(f'Base: {base}')

    for connection in root.findall('connection'):
        n = connection.get('name')
        logging.info(f'parsing connection: {n}')
        connection_beams = []
        connection_joints = []
        for element in connection.findall('element'):
            b = find_beam_by_name(element.get('component'), beams)
            connection_beams.append(b)
            connection_joints.append(b.get_component_by_name(element.get('joint')))
        connections.append(Connection(connection_beams[0], connection_joints[0], connection_beams[1], connection_joints[1]))

    if not isinstance(base,type(None)):
        return BeamAssembly(connections, base)
    else:
        raise ValueError("no base beam specified, one component beam must be set 'base=True'")

def load_beam_xml(beam_xml_file:str) -> list[Beam]:
    """loads beam structures from xml

    Args:
        beam_xml_file (str): file location

    Raises:
        ValueError: if beam component element is not specified correctly

    Returns:
        list[Beam]: set of beam descriptor objects

    example xml for a single beam, note multiple beams can be wrapped in the data tag.
    NOTE all beams, joints, and links must have unique names

    <data>
        <beam name="beam1">
            <joint name="J1" marker="1" part="in-f-end">
                <child link="L1"/>
            </joint>
            <link name="L1" length=100>
                <parent joint="J1"/>
                <child joint="J2"/>
            </link>
            <joint name="J2" marker="2" part="in-f-end">
                <parent link="L1"/>
            </joint>
        </beam>
    </data>

    valid type tags for joints:
        "in-m-end"
        "angle-m-end"
        "thru-f-end"
        "in-f-end"
        "thru-m"
        "thru-f"
        "angle-f"
        "in-f"
        
    end joints (with "-end") can carry markers for visual tracking.

    """
    logging.info(f"Processing beam file: {beam_xml_file}")
    beams = []
    tree = ET.parse(beam_xml_file, parser=ET.XMLParser(encoding='utf-8'))
    root = tree.getroot()
    possible_elements = ElementType.as_dict().keys()
    for beam in root: # loop through beams
        beam_name = beam.get('name')
        # loop though links
        beam_components = []
        for link in beam.findall('link'):
            link_name=link.get('name')
            if link_name == None:
                raise ValueError("all links and joints require 'name' tags")
            length = float(link.get('length'))
            if length == None:
                raise ValueError("LINKs must have a specified length")
            link_child = link.find('child').get('joint')
            link_parent = link.find('parent').get('joint')
            if isinstance(link_child, type(None)):
                raise ValueError("LINKs must have a parent and child joint")
            if isinstance(link_parent, type(None)):
                raise ValueError("LINKs must have a parent and child joint")
            beam_components.append({'name': link_name,
                                    'length': length,
                                    'child': link_child,
                                    'parent': link_parent,
                                    'type': ElementType.LINK,
                                    'marker':None
                                    })
        # loop through joints
        for joint in beam.findall('joint'):
            j_name=joint.get('name')
            if j_name == None:
                raise ValueError("all links and joints require 'name' tags")
            marker=joint.get('marker')
            j_child = joint.find('child')
            if not isinstance(j_child,type(None)):
                j_child_n = j_child.get("link")
            else: j_child_n = None
            j_parent = joint.find('parent')
            if not isinstance(j_parent,type(None)):
                j_parent_n=j_parent.get("link")
            else: j_parent_n = None
            try:
                t_code = joint.get('part')
            except Exception as e:
                raise ValueError("Joint must have a 'part' tag to define type")
            if (t_code not in possible_elements) or (t_code == "link"):
                raise ValueError("Joint specified with incorrect type")
            j_t = element_type_from_string(t_code)
            beam_components.append({'name': j_name,
                                    'length': None,
                                    'child': j_child_n,
                                    'parent': j_parent_n,
                                    'type':j_t,
                                    'marker':marker,})
        #logging.debug(beam_components)
        # create beam components
        components = [BeamComponent(co["name"],co["type"], child = None, parent=None,length=co["length"], marker=co["marker"]) for co in beam_components]

        def find_component_by_name(name:str, components:list[BeamComponent]):
            for e in components:
                if e.name == name: return e
            raise ValueError(f'No component {name} in beam components')

        # loop through componenets to create set of links and set of joints
        joints = []
        links = []
        for i in range(len(beam_components)):
            bc = components[i]
            comp = beam_components[i]
            # add parent or child data if valid
            if not isinstance(comp["child"], type(None)):
                bc.child = find_component_by_name(comp["child"], components)
            if not isinstance(comp["parent"],type(None)):
                bc.parent = find_component_by_name(comp["parent"], components)
            # add to correct list
            if comp["type"] == ElementType(0):
                links.append(bc)
            else:
                joints.append(bc)
        
        beams.append(Beam(beam_name, joints, links))
    #print(beams)
    return beams


def main():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    beams = load_beam_xml("example_beamset_latest.xml")
    # assem = load_assembly_xml(beams, "assembly_latest.xml")
    assem = load_assembly_xml(beams, "assembly_easy_2.xml")
    
    scene = assem.create_display_scene(colorise=True)
    scene.show()






if __name__ == "__main__":
    main()


