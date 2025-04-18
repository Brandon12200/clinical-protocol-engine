import json
import html
import logging
import re

logger = logging.getLogger(__name__)

class ResultsVisualization:
    """Visualization utilities for extraction results"""
    
    def highlight_entities(self, text, entities):
        """Add HTML highlighting for extracted entities in text"""
        try:
            # Sort entities by start position (reversed to avoid index issues when inserting HTML)
            sorted_entities = sorted(entities, key=lambda e: e['start'], reverse=True)
            
            # Convert text to HTML safe string
            html_text = html.escape(text)
            
            # Add highlighting for each entity
            for entity in sorted_entities:
                start = entity['start']
                end = entity['end']
                entity_type = entity['label']
                confidence = entity.get('confidence', 1.0)
                
                # Generate color based on entity type
                color = self._get_entity_color(entity_type)
                
                # Generate tooltip with entity type and confidence
                tooltip = f"{entity_type} (Confidence: {confidence:.2f})"
                
                # Insert highlighting HTML
                entity_text = html_text[start:end]
                highlight_html = f'<span class="entity" style="background-color: {color};" data-type="{entity_type}" data-confidence="{confidence}" title="{tooltip}">{entity_text}</span>'
                
                html_text = html_text[:start] + highlight_html + html_text[end:]
            
            # Convert newlines to HTML breaks
            html_text = html_text.replace('\n', '<br>')
            
            return html_text
        except Exception as e:
            logger.error(f"Error highlighting entities: {str(e)}")
            # Return original text if highlighting fails
            return html.escape(text).replace('\n', '<br>')
    
    def _get_entity_color(self, entity_type):
        """Get color for entity type"""
        color_map = {
            'ELIGIBILITY': '#ffcccb',  # Light red
            'PROCEDURE': '#c2f0c2',    # Light green
            'MEDICATION': '#c2e0ff',   # Light blue
            'ENDPOINT': '#ffffcc',     # Light yellow
            'TIMING': '#e6ccff',       # Light purple
            'DOSAGE': '#ffd8b1',       # Light orange
            'ADVERSE_EVENT': '#ffc0cb', # Pink
            'CONDITION': '#d3d3d3',    # Light gray
            'INCLUSION': '#ffe4b5',    # Moccasin
            'EXCLUSION': '#ffa07a'     # Light salmon
        }
        
        # Extract base entity type by removing B- or I- prefix
        if '-' in entity_type:
            base_type = entity_type.split('-')[1]
        else:
            base_type = entity_type
            
        return color_map.get(base_type, '#e0e0e0')  # Default to light gray
    
    def create_relation_graph(self, entities, relations):
        """Generate visualization data for entity relationships"""
        # Generate graph data structure for visualization library
        nodes = []
        edges = []
        
        # Create nodes from entities
        for i, entity in enumerate(entities):
            nodes.append({
                "id": i,
                "label": entity['text'],
                "group": entity['label'],
                "title": f"{entity['label']}: {entity['text']}"
            })
        
        # Create edges from relations
        for relation in relations:
            edges.append({
                "from": relation['source'],
                "to": relation['target'],
                "label": relation['type'],
                "title": f"{relation['type']} (Confidence: {relation.get('confidence', 1.0):.2f})"
            })
        
        return {
            "nodes": nodes,
            "edges": edges
        }
    
    def format_fhir_json(self, fhir_resources):
        """Create formatted display of FHIR resources"""
        # Format FHIR resources as syntax-highlighted JSON
        formatted_resources = {}
        
        for resource_type, resource in fhir_resources.items():
            # Pretty-print the JSON
            formatted_json = json.dumps(resource, indent=2)
            formatted_resources[resource_type] = formatted_json
        
        return formatted_resources
    
    def format_omop_tables(self, omop_data):
        """Create formatted display of OMOP data"""
        # Format each OMOP table
        formatted_tables = {}
        
        for table_name, records in omop_data.items():
            if not records:
                formatted_tables[table_name] = "No data"
                continue
                
            # Extract column names from first record
            columns = list(records[0].keys())
            
            # Format as HTML table
            html_table = "<table class='omop-table'><thead><tr>"
            
            # Add headers
            for column in columns:
                html_table += f"<th>{column}</th>"
            
            html_table += "</tr></thead><tbody>"
            
            # Add rows
            for record in records:
                html_table += "<tr>"
                for column in columns:
                    cell_value = record.get(column, "")
                    if cell_value is None:
                        cell_value = ""
                    html_table += f"<td>{html.escape(str(cell_value))}</td>"
                html_table += "</tr>"
            
            html_table += "</tbody></table>"
            formatted_tables[table_name] = html_table
        
        return formatted_tables
    
    def generate_confidence_visualization(self, entities):
        """Visualize confidence scores for extracted entities"""
        if not entities:
            return {"chart_data": [], "average_confidence": 0}
        
        # Group entities by type
        entity_types = {}
        for entity in entities:
            entity_type = entity['label']
            if '-' in entity_type:
                entity_type = entity_type.split('-')[1]
                
            if entity_type not in entity_types:
                entity_types[entity_type] = []
                
            entity_types[entity_type].append(entity['confidence'])
        
        # Calculate average confidence by type
        chart_data = []
        for entity_type, confidences in entity_types.items():
            avg_confidence = sum(confidences) / len(confidences)
            chart_data.append({
                "label": entity_type,
                "value": avg_confidence,
                "color": self._get_entity_color(entity_type),
                "count": len(confidences)
            })
        
        # Sort by confidence (descending)
        chart_data.sort(key=lambda x: x['value'], reverse=True)
        
        # Calculate overall average confidence
        all_confidences = [entity['confidence'] for entity in entities]
        average_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0
        
        return {
            "chart_data": chart_data,
            "average_confidence": average_confidence
        }
    
    def highlight_sections(self, text, sections):
        """Add HTML section dividers and headers to document text"""
        try:
            # Sort sections by start position (reversed to avoid index issues when inserting HTML)
            sorted_sections = sorted(sections, key=lambda s: s['start'], reverse=True)
            
            # Convert text to HTML safe string
            html_text = html.escape(text)
            
            # Add section dividers and headers
            for section in sorted_sections:
                start = section['start']
                end = section['end']
                section_type = section['type']
                
                # Create section header
                section_header = f'<div class="section-header" data-section-type="{section_type}">{section_type}</div>'
                section_start = f'<div class="section" data-section-type="{section_type}">'
                section_end = '</div>'
                
                # Insert section HTML
                html_text = html_text[:start] + section_header + section_start + html_text[start:end] + section_end + html_text[end:]
            
            # Convert newlines to HTML breaks
            html_text = html_text.replace('\n', '<br>')
            
            return html_text
        except Exception as e:
            logger.error(f"Error highlighting sections: {str(e)}")
            # Return original text if highlighting fails
            return html.escape(text).replace('\n', '<br>')
    
    def create_protocol_summary(self, protocol_data):
        """Generate a summary visualization of protocol elements"""
        summary = {
            "total_entities": 0,
            "entity_counts": {},
            "relation_counts": {},
            "average_confidence": 0,
            "section_counts": {}
        }
        
        # Count entities by type
        if 'entities' in protocol_data:
            entities = protocol_data['entities']
            summary["total_entities"] = len(entities)
            
            # Count by type and calculate average confidence
            confidences = []
            for entity in entities:
                entity_type = entity['label']
                if '-' in entity_type:
                    entity_type = entity_type.split('-')[1]
                
                if entity_type not in summary["entity_counts"]:
                    summary["entity_counts"][entity_type] = 0
                
                summary["entity_counts"][entity_type] += 1
                confidences.append(entity.get('confidence', 0))
            
            # Calculate average confidence
            if confidences:
                summary["average_confidence"] = sum(confidences) / len(confidences)
        
        # Count relations by type
        if 'relations' in protocol_data:
            relations = protocol_data['relations']
            for relation in relations:
                relation_type = relation['type']
                
                if relation_type not in summary["relation_counts"]:
                    summary["relation_counts"][relation_type] = 0
                
                summary["relation_counts"][relation_type] += 1
        
        # Count sections by type
        if 'sections' in protocol_data:
            sections = protocol_data['sections']
            for section in sections:
                section_type = section['type']
                
                if section_type not in summary["section_counts"]:
                    summary["section_counts"][section_type] = 0
                
                summary["section_counts"][section_type] += 1
        
        return summary
    
    def generate_side_by_side_comparison(self, original_text, highlighted_text, summary=None):
        """Generate HTML for side-by-side comparison of original and highlighted text"""
        html_output = """
        <div class="comparison-container">
            <div class="comparison-left">
                <h3>Original Document</h3>
                <div class="original-text">
                    {0}
                </div>
            </div>
            <div class="comparison-right">
                <h3>Extracted Elements</h3>
                <div class="highlighted-text">
                    {1}
                </div>
        """.format(
            html.escape(original_text).replace('\n', '<br>'),
            highlighted_text
        )
        
        # Add summary if provided
        if summary:
            html_output += """
                <div class="extraction-summary">
                    <h3>Extraction Summary</h3>
                    <p>Total Entities: {0}</p>
                    <p>Average Confidence: {1:.2f}</p>
                    <div class="entity-counts">
            """.format(
                summary.get("total_entities", 0),
                summary.get("average_confidence", 0)
            )
            
            # Add entity counts
            for entity_type, count in summary.get("entity_counts", {}).items():
                color = self._get_entity_color(entity_type)
                html_output += f'<div class="entity-count" style="border-left: 4px solid {color};"><span>{entity_type}</span>: {count}</div>'
            
            html_output += """
                    </div>
                </div>
            """
        
        html_output += """
            </div>
        </div>
        """
        
        return html_output