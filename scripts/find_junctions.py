import xml.etree.ElementTree as ET
from collections import Counter

def find_major_junctions(net_file, top_n=10):
    tree = ET.parse(net_file)
    root = tree.getroot()
    
    # Count incoming edges for each junction
    junction_incoming = Counter()
    for edge in root.findall('edge'):
        to_node = edge.get('to')
        if to_node:
            junction_incoming[to_node] += 1
            
    # Filter junctions that actually exist in the <junction> tags
    junction_types = {}
    for junction in root.findall('junction'):
        jid = junction.get('id')
        jtype = junction.get('type')
        if jtype != 'internal':
            # Extract raw IDs if it's a cluster
            if jid.startswith('cluster_'):
                raw_ids = jid.split('_')[1:]
                junction_types[jid] = (jtype, raw_ids)
            else:
                junction_types[jid] = (jtype, [jid])
            
    # Get top N junctions by incoming edge count
    major_junctions = []
    for jid, count in junction_incoming.most_common():
        if jid in junction_types:
            jtype, raw_ids = junction_types[jid]
            major_junctions.append((jid, count, jtype, raw_ids))
            if len(major_junctions) >= top_n:
                break
                
    return major_junctions

if __name__ == "__main__":
    net_file = "/home/kk/cap/data/raw/thoothukudi.net.xml"
    major = find_major_junctions(net_file)
    print("Top major junctions (including raw OSM IDs):")
    for jid, count, jtype, raw_ids in major:
        print(f"ID: {jid}, Incoming: {count}, Type: {jtype}, Raw IDs: {', '.join(raw_ids)}")
