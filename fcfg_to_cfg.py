import nltk
import sys
import re

def fcfg_to_cfg(fcfg_file, cfg_file, pcfg_file=None):
    """
    Convert a Finite Context-Free Grammar (FCFG) to a Context-Free Grammar (CFG).
    
    Parameters:
    fcfg_file (str): Path to the input FCFG file.
    cfg_file (str): Path to the output CFG file.a
    """
    # Read the FCFG from the file
    with open(fcfg_file, 'r') as f:
        fcfg_grammar = f.read()
    
    start_symbol = None
    productions = []
    for line in fcfg_grammar.splitlines():
        if line.strip() == "" or line.strip().startswith("#"):
            continue
          
        if line.strip().startswith("% start"):
            # Extract the start symbol from the line
            start_symbol = line.strip().split()[2].strip()
            continue
        
        if line.find("->") == -1:
            continue
       
        # Parse the FCFG grammar
        production = Production(line.strip())
        productions.append(production)
 
    # print("Parsed productions:")
    # for production in productions:
    #     # print(f"{production.lhs} -> {' '.join(production.rhs)}")
    #     print(f"symbol: {production.signature_node} features: {production.signature_node.features}, ")
    
    # Drop variable features in signature node that are not in the right nodes
    for production in productions:
        # Collect all feature values from all right nodes
        all_right_node_values = []
        for right_node in production.right_nodes:
            if right_node.features:
                for feature_dict in right_node.features:
                    all_right_node_values.extend(feature_dict.values())
        
        # Now check each feature in the signature node
        if production.signature_node.features:
            for feature_dict in production.signature_node.features:
                # Keys to remove (collect first to avoid modifying during iteration)
                keys_to_remove = []
                for key, value in list(feature_dict.items()):
                    # Only consider variable features (starting with ?)
                    if value.startswith('?'):
                        # Remove if this variable value is not found in any right node
                        if value not in all_right_node_values:
                            keys_to_remove.append(key)
                
                # Remove the keys that weren't found in any right node
                for key in keys_to_remove:
                    del feature_dict[key]
                    
                
    # Create a mapping structure for signature symbol, feature set index, and key to their possible values
    # This allows handling multiple feature dictionaries from slash notation
    sign_and_feature_to_values = {}
    
    for production in productions:
        # Loop through each feature dictionary in the signature node's features list
        for feature_idx, feature_dict in enumerate(production.signature_node.features):
            for key, value in feature_dict.items():
                if not value.startswith('?'):
                    # Create a composite key: (symbol, feature_set_index, feature_key)
                    composite_key = (production.signature_node, feature_idx, key)
                    
                    if composite_key not in sign_and_feature_to_values:
                        sign_and_feature_to_values[composite_key] = set()
                    
                    sign_and_feature_to_values[composite_key].add(value)
                    
    # print("Initial sign_and_feature_to_values mapping:")
    # for (node, index, key), values in sign_and_feature_to_values.items():
    #     print(f"Node: {node}, Index: {index}, Key: {key}, Values: {values}")
                    
    # Fill the final mapping with all possible values for each feature
    changed = True
    while changed:
        changed = False
        for production in productions:
            # Iterate through each feature dictionary in the signature node
            for feature_idx, feature_dict in enumerate(production.signature_node.features):
                for key, value in feature_dict.items():
                    if not value.startswith('?'):
                        continue
                    
                    # We need to find nodes that have this variable value
                    variable_matches = []
                    
                    # Check all right nodes for this variable
                    for right_node in production.right_nodes:
                        # For each right node, check all its feature dictionaries
                        for r_feature_idx, r_feature_dict in enumerate(right_node.features):
                            # Check if this feature dictionary contains our variable
                            # Iterate through all key-value pairs looking for the variable value
                            for r_key, r_value in r_feature_dict.items():
                                if r_value == value:  # Match on variable name, not key
                                    # Found a match - store node, feature set index, and key
                                    variable_matches.append((right_node, r_feature_idx, r_key))
                    
                    if not variable_matches:
                        continue
                    
                    # Get possible values for each matching node+feature combination
                    mapped_possible_values = []
                    for match in variable_matches:
                        # Use the composite key to get values
                        possible_values = sign_and_feature_to_values.get(match, set())
                        if possible_values:  # Only add non-empty sets
                            mapped_possible_values.append(possible_values)
                    
                    # Skip if we don't have any values or if any set is empty
                    if not mapped_possible_values or any(len(values) == 0 for values in mapped_possible_values):
                        continue
                    
                    # Intersect the possible values for this feature across all nodes
                    possible_values = set.intersection(*mapped_possible_values)
                    
                    # Skip if the intersection is empty
                    if not possible_values:
                        continue
                    
                    # Create the composite key for the current node and feature
                    current_key = (production.signature_node, feature_idx, key)
                    
                    # Check if the possible values are already in the map
                    if sign_and_feature_to_values.get(current_key) == possible_values:
                        continue
                    
                    # Update the mapping with the new possible values
                    sign_and_feature_to_values[current_key] = possible_values
                    changed = True
    
    print("Final sign_and_feature_to_values mapping:")
    for (node, index, key), values in sign_and_feature_to_values.items():
        print(f"Node: {node}, Index: {index}, Key: {key}, Values: {values}")
    
    # Test all productions with ? features are in the signature and feature to values mapping
    # and test all nodes of a production have a signature and feature to values mapping
    convertible_productions = []

    for production in productions:
        # Check production features first
        skip_production = False
        
        # Collect all variable features on the left-hand side
        lhs_variables = set()
        
        # Iterate through each feature dictionary in the signature node
        for feature_idx, feature_dict in enumerate(production.signature_node.features):
            for key, value in feature_dict.items():
                if value.startswith('?'):
                    # Add to the set of LHS variables
                    lhs_variables.add(value)
                    
                    # Check if the feature has values in the mapping
                    composite_key = (production.signature_node, feature_idx, key)
                    if composite_key not in sign_and_feature_to_values or \
                    len(sign_and_feature_to_values.get(composite_key, set())) == 0:
                        print(f"Warning: Production '{production.lhs} -> {' '.join(production.rhs)}' "
                            f"cannot be properly converted because no values were found for feature {key} "
                            f"in feature set {feature_idx}.\n")
                        skip_production = True
                        break
            
            # Skip to next production if needed
            if skip_production:
                break
        
        if skip_production:
            continue
        
        # Collect all variable features on the right-hand side
        rhs_variables = set()
        
        # Now check all nodes in the production
        for node in production.right_nodes:
            # Iterate through each feature dictionary in the right node
            for feature_idx, feature_dict in enumerate(node.features):
                for key, value in feature_dict.items():
                    if value.startswith('?'):
                        # Add to the set of RHS variables
                        rhs_variables.add(value)
                        
                        # Check if the feature has values in the mapping
                        composite_key = (node, feature_idx, key)
                        if composite_key not in sign_and_feature_to_values or \
                        len(sign_and_feature_to_values.get(composite_key, set())) == 0:
                            print(f"Warning: Node '{node}' in production '{production.lhs} -> {' '.join(production.rhs)}' "
                                f"cannot be properly converted because no values were found for feature {key} "
                                f"in feature set {feature_idx}.\n")
                            skip_production = True
                            break
                
                if skip_production:
                    break
            
            if skip_production:
                break
        
        # Check if all LHS variables appear in the RHS
        if not skip_production:
            ungrounded_vars = lhs_variables - rhs_variables
            if ungrounded_vars:
                var_list = ", ".join(sorted(ungrounded_vars))
                print(f"Warning: Production '{production.lhs} -> {' '.join(production.rhs)}' "
                    f"has ungrounded variables on the left-hand side: {var_list}. "
                    f"These variables do not appear on the right-hand side.\n")
                skip_production = True
        
        if not skip_production:
            convertible_productions.append(production)
    
    # Now we can convert the productions to CFG format
    cfg_productions = convert_productions_with_features(convertible_productions, sign_and_feature_to_values)
    # Remove duplicates but preserving order
    # cfg_productions = list(set(cfg_productions))
    seen = set()
    cfg_productions = [x for x in cfg_productions if not (x in seen or seen.add(x))]
    
    
    # print("CFG Productions:")
    # for cfg_production in cfg_productions:
    #     print(cfg_production)
        
    # Output the CFG to the specified file
    with open(cfg_file, 'w') as f:
        if start_symbol:
            f.write(f"% start {start_symbol}\n")
        for cfg_production in cfg_productions:
            f.write(cfg_production + "\n")
       
    lhs_dict = {}     
    for cfg_production in cfg_productions:
        ## split by the one arrow into lhs and rhs
        lhs, rhs = cfg_production.split("->")
        lhs = lhs.strip()
        if lhs not in lhs_dict:
            lhs_dict[lhs] = 0
        lhs_dict[lhs] += 1
        
    # convert to pcfg with equal probabilities
    pcfg_productions = []
    for cfg_production in cfg_productions:
        lhs, rhs = cfg_production.split("->")
        lhs = lhs.strip()
        pcfg_productions.append(f"{lhs} -> {rhs.strip()} [{1/lhs_dict[lhs]}]")
    
    # print("PCFG Productions:")
    # for pcfg_production in pcfg_productions:
    #     print(pcfg_production)
        
    if pcfg_file:
        with open(pcfg_file, 'w') as f:
            if start_symbol:
                f.write(f"% start {start_symbol}\n")
            for pcfg_production in pcfg_productions:
                f.write(pcfg_production + "\n")
            
def convert_productions_with_features(convertible_productions, sign_and_feature_to_values):
    cfg_productions = []
    
    for production in convertible_productions:
        # Find all features that start with '?' and their possible values
        variable_features = {}
        
        # Check signature node for variables
        for feature_idx, feature_dict in enumerate(production.signature_node.features):
            for key, value in feature_dict.items():
                if value.startswith('?'):
                    var_name = value  # Keep the full variable name including '?'
                    composite_key = (production.signature_node, feature_idx, key)
                    possible_values = sign_and_feature_to_values.get(composite_key, set())
                    
                    if len(possible_values) == 0:
                        raise ValueError(f"No possible values found for feature '{key}' in production signature node '{production.signature_node.symbol}'.")
                    
                    # Group by variable name to tie features together across the entire production
                    if var_name not in variable_features:
                        variable_features[var_name] = []
                    
                    variable_features[var_name].append((composite_key, possible_values))
        
        # Check all right nodes for variables
        for node in production.right_nodes:
            for feature_idx, feature_dict in enumerate(node.features):
                for key, value in feature_dict.items():
                    if value.startswith('?'):
                        var_name = value  # Keep the full variable name including '?'
                        composite_key = (node, feature_idx, key)
                        possible_values = sign_and_feature_to_values.get(composite_key, set())
                        
                        if len(possible_values) == 0:
                            raise ValueError(f"No possible values found for feature '{key}' in node '{node}'.")
                        
                        # Group by variable name to tie features together across the entire production
                        if var_name not in variable_features:
                            variable_features[var_name] = []
                        
                        variable_features[var_name].append((composite_key, possible_values))
        
        # Generate all combinations of feature values
        value_combinations = generate_value_combinations(variable_features)
        
        # Create a new production for each combination
        for combination in value_combinations:
            # Create string representation of the production with specific feature values
            lhs = generate_node_string(production.signature_node, combination)
            rhs = [generate_node_string(node, combination) for node in production.right_nodes]
            
            # Add the new production string to cfg_productions
            cfg_production = f"{lhs} -> {' '.join(rhs)}"
            cfg_productions.append(cfg_production)
    
    return cfg_productions

def generate_value_combinations(variable_features):
    """
    Generate all combinations of feature values.
    Each variable must have the same value across the entire production.
    """
    if not variable_features:
        return [{}]
    
    result = [{}]
    
    # For each variable, find the intersection of possible values across all occurrences
    for var_name, feature_entries in variable_features.items():
        # Extract all sets of possible values for this variable
        value_sets = [possible_values for _, possible_values in feature_entries]
        
        # Find the intersection of all possible values for this variable
        common_values = set.intersection(*value_sets) if value_sets else set()
        
        if not common_values:
            raise ValueError(f"No common values found for variable '{var_name}' across all its occurrences.")
        
        # Generate new combinations with each possible value
        new_result = []
        for combination in result:
            for value in common_values:
                new_combination = combination.copy()
                
                # Assign the same value to all features tied to this variable
                for composite_key, _ in feature_entries:
                    new_combination[composite_key] = value
                
                new_result.append(new_combination)
        
        result = new_result
    
    return result

def generate_node_string(node, feature_values):
    """
    Generate a string representation of a node with specific feature values.
    For symbols with slashes, each feature dictionary corresponds to the part of the symbol at the same index.
    
    Examples:
    - For symbol "NP" with features [{"NUM": "sg", "PERS": "3"}], result: "NP-sg-3"
    - For symbol "VP/NP" with features [{"TENSE": "past"}, {"NUM": "pl"}], result: "VP-past/NP-pl"
    - For symbol "S/NP/VP" with features [{"NUM": "sg"}, {"CASE": "acc"}, {"TENSE": "pres"}], 
      result: "S-sg/NP-acc/VP-pres"
    """
    # Split the symbol by slash
    symbol_parts = node.symbol.split('/')
    
    # Process each part individually
    processed_parts = []
    
    for part_idx, part in enumerate(symbol_parts):
        # Get the corresponding feature dictionary if available
        if part_idx < len(node.features):
            feature_dict = node.features[part_idx]
            
            # Collect feature values for this part
            part_features = []
            
            # Sort keys to ensure consistent order
            sorted_keys = sorted(feature_dict.keys())
            
            for key in sorted_keys:
                value = feature_dict[key]
                
                # Skip semantic features
                if key.lower() == "sem":
                    continue
                    
                if value.startswith('?'):
                    # Use the specific value from the combination
                    composite_key = (node, part_idx, key)
                    if composite_key not in feature_values:
                        raise ValueError(f"Missing value for feature '{key}' of node '{node}' with variable '{value}'")
                    feature_value = feature_values[composite_key]
                    part_features.append(feature_value)
                else:
                    part_features.append(value)
            
            # Append features to this part
            if part_features:
                part += '-' + '-'.join(part_features)
        
        processed_parts.append(part)
    
    # Join the processed parts back with slashes
    result = '/'.join(processed_parts)
    
    return result

class Production:
    def __init__(self, line):
        self.lhs, self.rhs = self.parse_production(line)
        self.signature_node = Node(self.lhs)
        self.right_nodes = [Node(signature) for signature in self.rhs]
      
    def parse_production(self, line: str) -> tuple:
        """
        Parse a single production line from the FCFG.
        Parameters:
        line (str): A line from the FCFG file.
        Returns:
        tuple: A tuple containing the left-hand side and right-hand side of the production.
        """
        # Check if the line is empty or a comment
        if not line or line.strip().startswith('#'):
            return None, None
            
        # Parse considering brackets to find the actual production arrow
        bracket_count = 0
        split_pos = -1
        
        for i in range(len(line) - 1):
            if line[i:i+2] == "->" and bracket_count == 0:
                split_pos = i
                break
            elif line[i] == '[':
                bracket_count += 1
            elif line[i] == ']':
                bracket_count -= 1
        
        # If no production arrow found outside brackets
        if split_pos == -1:
            return None, None
            
        # Split the line at the correct position
        lhs = line[:split_pos].strip()
        rhs = line[split_pos+2:].strip()  # +2 to skip the "->"
        
        # Handle empty right-hand side
        if not rhs:
            return lhs, []
        
        # Parse the right-hand side considering complex productions
        productions = []
        current_token = ""
        in_feature = False
        bracket_count = 0
        
        for char in rhs + " ":
            if char == "[":
                in_feature = True
                bracket_count += 1
                current_token += char
            elif char == "]":
                in_feature = False
                bracket_count -= 1
                current_token += char
            elif char.isspace() and not in_feature and bracket_count == 0:
                if current_token:  # Only add non-empty tokens
                    productions.append(current_token)
                    current_token = ""
            else:
                current_token += char
        
        return lhs, productions

    
class Node:
    def __init__(self, signature: str):
        self.is_terminal = self._is_terminal(signature)
        self.symbol = self._extract_symbol(signature)
        self.features = self._get_features(signature)
        self.feature_pairs = []
        for feature_dict in self.features:
            for key, value in feature_dict.items():
                self.feature_pairs.append((key, value))
        
        self.all_keys = [k for k, _ in self.feature_pairs]
        self.all_values = [v for _, v in self.feature_pairs]
        
    def _extract_symbol(self, signature):
        # For slash notation, we need to extract symbols from both sides
        if '/' in signature:
            parts = signature.split('/')
            symbols = []
            for part in parts:
                # Extract symbol (everything before '[' if it exists)
                if '[' in part:
                    symbols.append(part.split('[', 1)[0])
                else:
                    symbols.append(part)
            return '/'.join(symbols)
        
        # Simple case - no slash
        if '[' in signature:
            return signature.split('[', 1)[0]
        return signature
    
    def _is_terminal(self, signature: str) -> bool:
        # A terminal symbol is a string characters wrapped in quotes
        pattern = r"^['\"].*['\"]$"
        return bool(re.match(pattern, signature.strip()))
      
    def _get_features(self, signature: str) -> list:
        feature_sets = []
        
        # Handle signature with slash notation
        if '/' in signature:
            parts = signature.split('/')
            for part in parts:
                features = self._extract_features_from_part(part)
                if features:
                    feature_sets.append(features)
        else:
            # Simple case - just one set of features
            features = self._extract_features_from_part(signature)
            if features:
                feature_sets.append(features)
        
        return feature_sets

    def _extract_features_from_part(self, part: str) -> dict:
        features = {}
        # Get string inside the brackets if they exist
        bracket_match = re.search(r'\[(.*?)\]', part.strip())
        if bracket_match:
            features_str = bracket_match.group(1)
            
            # Use a more sophisticated approach to split by commas outside angle brackets
            feature_parts = []
            current_part = ""
            angle_bracket_level = 0
            
            for char in features_str:
                if char == '<':
                    angle_bracket_level += 1
                    current_part += char
                elif char == '>':
                    angle_bracket_level -= 1
                    current_part += char
                elif char == ',' and angle_bracket_level == 0:
                    # Only split on commas outside angle brackets
                    feature_parts.append(current_part.strip())
                    current_part = ""
                else:
                    current_part += char
                    
            # Add the last part if it exists
            if current_part.strip():
                feature_parts.append(current_part.strip())
            
            # Process each feature part
            for feature in feature_parts:
                key_value = feature.strip().split('=', 1)  # Split on first '=' only
                if len(key_value) == 2:
                    key = key_value[0].strip()
                    value = key_value[1].strip()
                    
                    # Skip semantic features if needed
                    if key.lower() == "sem":
                        continue
                        
                    # Value can be wrapped in <> or not, so we need to handle that
                    if value.startswith('<') and value.endswith('>'):
                        value = value[1:-1].strip()
                        
                    features[key] = value
                    
        return features
      
    def __eq__(self, value):
        if not isinstance(value, Node):
            print("Not a Node instance")
            return False
            
        # Check if symbols match
        if self.symbol != value.symbol:
            return False
            
        # Check if they have the same number of feature sets
        if len(self.features) != len(value.features):
            return False
            
        # For each feature set, check if they have the same keys
        for i, feature_set in enumerate(self.features):
            # If we've gone beyond the number of feature sets in the other node
            if i >= len(value.features):
                return False
                
            # Compare keys in each feature set
            if set(feature_set.keys()) != set(value.features[i].keys()):
                return False
                
        return True

    def __hash__(self):
        # Create a hashable representation of feature sets (keys only)
        feature_sets_hash = tuple(frozenset(feature_set.keys()) for feature_set in self.features)
        return hash((self.symbol, feature_sets_hash))

    def __repr__(self):
        return f"Node(symbol={self.symbol}, features={self.features})"
    
            
# if __name__ == "__main__":
#   # Check if the correct number of arguments is provided
#   if len(sys.argv) != 3:
#     print("Usage: python fcfg-to-cfg.py <fcfg_file_name> <output_file_name>")
#     sys.exit(1)
  
#   # Get file paths from command line arguments
#   fcfg_file = sys.argv[1]
#   cfg_file = sys.argv[2]
  
#   # Call the conversion function
#   fcfg_to_cfg(fcfg_file, cfg_file)
#   print(f"Conversion complete. CFG written to {cfg_file}")

if __name__ == "__main__":
  # Check if the correct number of arguments is provided
  fcfg_to_cfg("ps6_grammar.fcfg", "cfg_grammer.cfg", "pcfg_grammer.pcfg")