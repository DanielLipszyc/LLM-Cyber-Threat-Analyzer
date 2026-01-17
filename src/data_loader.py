"""
Data loader for threat intelligence sources.
Supports CVE data, MITRE ATT&CK, and custom JSON files.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import requests

from .config import settings


@dataclass
class Document:
    """Represents a threat intelligence document."""
    id: str
    title: str
    content: str
    source: str
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "content": self.content,
            "source": self.source,
            "metadata": self.metadata
        }


class DataLoader:
    """Load and process threat intelligence data from various sources."""
    
    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or settings.data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def load_json_file(self, filepath: Path) -> List[Document]:
        """Load documents from a JSON file."""
        with open(filepath, "r") as f:
            data = json.load(f)
        
        documents = []
        items = data if isinstance(data, list) else [data]
        
        for item in items:
            doc = Document(
                id=item.get("id", f"doc_{len(documents)}"),
                title=item.get("title", "Untitled"),
                content=item.get("content", ""),
                source=item.get("source", filepath.stem),
                metadata=item.get("metadata", {})
            )
            documents.append(doc)
        
        return documents
    
    def load_cve_data(self, filepath: Optional[Path] = None) -> List[Document]:
        """Load CVE data from JSON file."""
        filepath = filepath or self.data_dir / "cve_sample.json"
        
        if not filepath.exists():
            print(f"CVE data file not found: {filepath}")
            return []
        
        with open(filepath, "r") as f:
            data = json.load(f)
        
        documents = []
        cves = data.get("cves", data) if isinstance(data, dict) else data
        
        for cve in cves:
            cve_id = cve.get("id", cve.get("cve_id", ""))
            description = cve.get("description", cve.get("descriptions", [{}]))
            
            if isinstance(description, list):
                description = " ".join([d.get("value", "") for d in description])
            
            content = f"""
CVE ID: {cve_id}

Description: {description}

Severity: {cve.get("severity", cve.get("cvss", {}).get("severity", "Unknown"))}
CVSS Score: {cve.get("cvss_score", cve.get("cvss", {}).get("score", "N/A"))}

Affected Products: {cve.get("affected_products", "See vendor advisories")}

References: {cve.get("references", [])}

Published: {cve.get("published", cve.get("published_date", "Unknown"))}
""".strip()
            
            doc = Document(
                id=cve_id,
                title=f"{cve_id}: {description[:100]}...",
                content=content,
                source="CVE Database",
                metadata={
                    "type": "cve",
                    "severity": cve.get("severity", "Unknown"),
                    "cvss_score": cve.get("cvss_score", "N/A"),
                    "published": cve.get("published", "Unknown")
                }
            )
            documents.append(doc)
        
        return documents
    
    def load_mitre_attack(self, filepath: Optional[Path] = None) -> List[Document]:
        """Load MITRE ATT&CK technique data."""
        filepath = filepath or self.data_dir / "mitre_attack.json"
        
        if not filepath.exists():
            print(f"MITRE ATT&CK data file not found: {filepath}")
            return []
        
        with open(filepath, "r") as f:
            data = json.load(f)
        
        documents = []
        techniques = data.get("techniques", data) if isinstance(data, dict) else data
        
        for technique in techniques:
            tech_id = technique.get("id", technique.get("technique_id", ""))
            name = technique.get("name", "")
            description = technique.get("description", "")
            
            content = f"""
MITRE ATT&CK Technique: {tech_id} - {name}

Description: {description}

Tactic: {technique.get("tactic", "Unknown")}
Platform: {technique.get("platforms", ["Unknown"])}

Detection: {technique.get("detection", "See MITRE ATT&CK documentation")}

Mitigation: {technique.get("mitigation", "See MITRE ATT&CK documentation")}

Examples: {technique.get("examples", [])}
""".strip()
            
            doc = Document(
                id=tech_id,
                title=f"{tech_id}: {name}",
                content=content,
                source="MITRE ATT&CK",
                metadata={
                    "type": "mitre_attack",
                    "tactic": technique.get("tactic", "Unknown"),
                    "platforms": technique.get("platforms", [])
                }
            )
            documents.append(doc)
        
        return documents
    
    def load_all_data(self) -> List[Document]:
        """Load all available threat intelligence data."""
        documents = []
        
        # Load CVE data
        documents.extend(self.load_cve_data())
        
        # Load MITRE ATT&CK data
        documents.extend(self.load_mitre_attack())
        
        # Load any additional JSON files in data directory
        for json_file in self.data_dir.glob("*.json"):
            if json_file.stem not in ["cve_sample", "mitre_attack"]:
                try:
                    documents.extend(self.load_json_file(json_file))
                except Exception as e:
                    print(f"Error loading {json_file}: {e}")
        
        print(f"Loaded {len(documents)} documents total")
        return documents


def fetch_recent_cves(limit: int = 100) -> List[Dict]:
    """
    Fetch recent CVEs from NVD API.
    Note: In production, you'd want to handle rate limiting and pagination.
    """
    # This is a simplified example - NVD API requires API key for higher rate limits
    url = f"https://services.nvd.nist.gov/rest/json/cves/2.0?resultsPerPage={limit}"
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()
        return data.get("vulnerabilities", [])
    except Exception as e:
        print(f"Error fetching CVEs: {e}")
        return []
