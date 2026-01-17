#!/usr/bin/env python3
"""
Fetch real CVE data from the National Vulnerability Database (NVD) API.

Usage:
    python scripts/fetch_nvd_cves.py                    # Fetch 100 recent CVEs
    python scripts/fetch_nvd_cves.py --count 500        # Fetch 500 CVEs
    python scripts/fetch_nvd_cves.py --year 2024        # Fetch CVEs from 2024
    python scripts/fetch_nvd_cves.py --keyword apache   # Search for Apache CVEs
    python scripts/fetch_nvd_cves.py --severity CRITICAL # Only critical CVEs

Note: NVD API is free but rate-limited.
    - Without API key: 5 requests per 30 seconds
    - With API key: 50 requests per 30 seconds
    
Get a free API key at: https://nvd.nist.gov/developers/request-an-api-key
"""

import argparse
import json
import time
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

import requests

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class NVDFetcher:
    """Fetch CVE data from the National Vulnerability Database API."""
    
    BASE_URL = "https://services.nvd.nist.gov/rest/json/cves/2.0"
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key
        self.headers = {}
        if api_key:
            self.headers["apiKey"] = api_key
        
        # Rate limiting
        self.request_delay = 6.0 if not api_key else 0.6  # seconds between requests
        self.last_request_time = 0
    
    def _wait_for_rate_limit(self):
        """Wait to respect rate limits."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.request_delay:
            time.sleep(self.request_delay - elapsed)
        self.last_request_time = time.time()
    
    def _make_request(self, params: Dict[str, Any]) -> Dict:
        """Make a rate-limited request to the NVD API."""
        self._wait_for_rate_limit()
        
        try:
            response = requests.get(
                self.BASE_URL,
                params=params,
                headers=self.headers,
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"API request failed: {e}")
            return {"vulnerabilities": []}
    
    def fetch_cves(
        self,
        count: int = 100,
        keyword: str = None,
        severity: str = None,
        year: int = None,
        days_back: int = None
    ) -> List[Dict[str, Any]]:
        """
        Fetch CVEs from NVD API.
        
        Args:
            count: Number of CVEs to fetch (max per request is 2000)
            keyword: Search keyword (e.g., "apache", "windows")
            severity: Filter by severity (CRITICAL, HIGH, MEDIUM, LOW)
            year: Filter by year published
            days_back: Fetch CVEs from the last N days
        
        Returns:
            List of CVE dictionaries
        """
        all_cves = []
        start_index = 0
        results_per_page = min(count, 2000)
        
        # Build base params
        params = {
            "resultsPerPage": results_per_page,
            "startIndex": start_index
        }
        
        # Add filters
        if keyword:
            params["keywordSearch"] = keyword
        
        if severity:
            params["cvssV3Severity"] = severity.upper()
        
        if year:
            params["pubStartDate"] = f"{year}-01-01T00:00:00.000"
            params["pubEndDate"] = f"{year}-12-31T23:59:59.999"
        elif days_back:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            params["pubStartDate"] = start_date.strftime("%Y-%m-%dT00:00:00.000")
            params["pubEndDate"] = end_date.strftime("%Y-%m-%dT23:59:59.999")
        
        print(f"Fetching CVEs from NVD API...")
        print(f"Filters: keyword={keyword}, severity={severity}, year={year}, days_back={days_back}")
        
        while len(all_cves) < count:
            params["startIndex"] = start_index
            params["resultsPerPage"] = min(results_per_page, count - len(all_cves))
            
            print(f"  Fetching {params['resultsPerPage']} CVEs starting at index {start_index}...")
            
            data = self._make_request(params)
            vulnerabilities = data.get("vulnerabilities", [])
            
            if not vulnerabilities:
                print("  No more results available.")
                break
            
            all_cves.extend(vulnerabilities)
            start_index += len(vulnerabilities)
            
            total_results = data.get("totalResults", 0)
            print(f"  Retrieved {len(all_cves)}/{min(count, total_results)} CVEs")
            
            if len(vulnerabilities) < params["resultsPerPage"]:
                break
        
        return all_cves[:count]
    
    def parse_cve(self, vuln: Dict) -> Dict[str, Any]:
        """Parse a CVE vulnerability object into our format."""
        cve = vuln.get("cve", {})
        cve_id = cve.get("id", "")
        
        # Get description (English)
        descriptions = cve.get("descriptions", [])
        description = ""
        for desc in descriptions:
            if desc.get("lang") == "en":
                description = desc.get("value", "")
                break
        
        # Get CVSS score and severity
        cvss_score = None
        severity = None
        
        metrics = cve.get("metrics", {})
        
        # Try CVSS v3.1 first, then v3.0, then v2
        for cvss_version in ["cvssMetricV31", "cvssMetricV30", "cvssMetricV2"]:
            if cvss_version in metrics and metrics[cvss_version]:
                cvss_data = metrics[cvss_version][0]
                if "cvssData" in cvss_data:
                    cvss_score = cvss_data["cvssData"].get("baseScore")
                    severity = cvss_data["cvssData"].get("baseSeverity")
                    if not severity:
                        severity = cvss_data.get("baseSeverity")
                    break
        
        # Get affected products (CPE)
        affected_products = []
        configurations = cve.get("configurations", [])
        for config in configurations:
            for node in config.get("nodes", []):
                for cpe_match in node.get("cpeMatch", []):
                    criteria = cpe_match.get("criteria", "")
                    # Extract product name from CPE string
                    # Format: cpe:2.3:a:vendor:product:version:...
                    parts = criteria.split(":")
                    if len(parts) >= 5:
                        vendor = parts[3]
                        product = parts[4]
                        affected_products.append(f"{vendor} {product}")
        
        # Deduplicate and limit
        affected_products = list(set(affected_products))[:10]
        
        # Get references
        references = []
        for ref in cve.get("references", [])[:5]:
            references.append(ref.get("url", ""))
        
        # Get dates
        published = cve.get("published", "")[:10]  # Just the date part
        
        # Get weaknesses (CWE)
        weaknesses = []
        for weakness in cve.get("weaknesses", []):
            for desc in weakness.get("description", []):
                if desc.get("lang") == "en":
                    weaknesses.append(desc.get("value", ""))
        
        return {
            "id": cve_id,
            "title": f"{cve_id}: {description[:100]}..." if len(description) > 100 else f"{cve_id}: {description}",
            "description": description,
            "severity": severity,
            "cvss_score": cvss_score,
            "affected_products": ", ".join(affected_products) if affected_products else "See vendor advisories",
            "references": references,
            "published": published,
            "weaknesses": weaknesses,
            "content": self._format_content(cve_id, description, severity, cvss_score, affected_products, references, published, weaknesses)
        }
    
    def _format_content(
        self,
        cve_id: str,
        description: str,
        severity: str,
        cvss_score: float,
        affected_products: List[str],
        references: List[str],
        published: str,
        weaknesses: List[str]
    ) -> str:
        """Format CVE data into readable content for the RAG system."""
        content = f"""CVE ID: {cve_id}

Description: {description}

Severity: {severity or 'Unknown'}
CVSS Score: {cvss_score or 'N/A'}

Affected Products: {', '.join(affected_products) if affected_products else 'See vendor advisories'}

Weaknesses (CWE): {', '.join(weaknesses) if weaknesses else 'Not specified'}

References: {', '.join(references) if references else 'See NVD for details'}

Published: {published}

For more information, see: https://nvd.nist.gov/vuln/detail/{cve_id}
"""
        return content.strip()


def fetch_mitre_attack() -> List[Dict[str, Any]]:
    """
    Fetch MITRE ATT&CK techniques from their GitHub repository.
    """
    print("Fetching MITRE ATT&CK data...")
    
    url = "https://raw.githubusercontent.com/mitre/cti/master/enterprise-attack/enterprise-attack.json"
    
    try:
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        print(f"Failed to fetch MITRE ATT&CK data: {e}")
        return []
    
    techniques = []
    
    for obj in data.get("objects", []):
        if obj.get("type") != "attack-pattern":
            continue
        
        # Get technique ID
        external_refs = obj.get("external_references", [])
        technique_id = ""
        for ref in external_refs:
            if ref.get("source_name") == "mitre-attack":
                technique_id = ref.get("external_id", "")
                break
        
        if not technique_id:
            continue
        
        name = obj.get("name", "")
        description = obj.get("description", "")
        
        # Get kill chain phases (tactics)
        tactics = []
        for phase in obj.get("kill_chain_phases", []):
            if phase.get("kill_chain_name") == "mitre-attack":
                tactics.append(phase.get("phase_name", ""))
        
        # Get platforms
        platforms = obj.get("x_mitre_platforms", [])
        
        # Get detection
        detection = obj.get("x_mitre_detection", "See MITRE ATT&CK documentation")
        
        content = f"""MITRE ATT&CK Technique: {technique_id} - {name}

Description: {description}

Tactic(s): {', '.join(tactics) if tactics else 'Unknown'}
Platform(s): {', '.join(platforms) if platforms else 'Multiple'}

Detection: {detection[:500] if detection else 'See MITRE ATT&CK documentation'}

For more information, see: https://attack.mitre.org/techniques/{technique_id.replace('.', '/')}/
"""
        
        techniques.append({
            "id": technique_id,
            "name": name,
            "title": f"{technique_id}: {name}",
            "description": description,
            "tactic": ", ".join(tactics),
            "platforms": platforms,
            "detection": detection,
            "content": content.strip(),
            "source": "MITRE ATT&CK",
            "metadata": {
                "type": "mitre_attack",
                "tactic": ", ".join(tactics),
                "platforms": platforms
            }
        })
    
    print(f"Fetched {len(techniques)} MITRE ATT&CK techniques")
    return techniques


def main():
    parser = argparse.ArgumentParser(description="Fetch CVE data from NVD API")
    parser.add_argument("--count", type=int, default=100, help="Number of CVEs to fetch")
    parser.add_argument("--keyword", type=str, help="Search keyword (e.g., 'apache', 'windows')")
    parser.add_argument("--severity", type=str, choices=["CRITICAL", "HIGH", "MEDIUM", "LOW"], help="Filter by severity")
    parser.add_argument("--year", type=int, help="Filter by year published")
    parser.add_argument("--days", type=int, help="Fetch CVEs from the last N days")
    parser.add_argument("--api-key", type=str, help="NVD API key (optional, increases rate limit)")
    parser.add_argument("--output", type=str, default="data/nvd_cves.json", help="Output file path")
    parser.add_argument("--include-mitre", action="store_true", help="Also fetch MITRE ATT&CK data")
    parser.add_argument("--ingest", action="store_true", help="Automatically ingest into RAG system")
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Fetch CVEs
    fetcher = NVDFetcher(api_key=args.api_key)
    raw_cves = fetcher.fetch_cves(
        count=args.count,
        keyword=args.keyword,
        severity=args.severity,
        year=args.year,
        days_back=args.days
    )
    
    # Parse CVEs into our format
    cves = [fetcher.parse_cve(vuln) for vuln in raw_cves]
    
    print(f"\nParsed {len(cves)} CVEs")
    
    # Save CVEs
    cve_output = {
        "source": "NVD",
        "fetched_at": datetime.now().isoformat(),
        "filters": {
            "keyword": args.keyword,
            "severity": args.severity,
            "year": args.year,
            "days_back": args.days
        },
        "cves": cves
    }
    
    with open(output_path, "w") as f:
        json.dump(cve_output, f, indent=2)
    
    print(f"Saved CVEs to {output_path}")
    
    # Fetch MITRE ATT&CK if requested
    if args.include_mitre:
        techniques = fetch_mitre_attack()
        mitre_output_path = output_path.parent / "mitre_attack_full.json"
        
        mitre_output = {
            "source": "MITRE ATT&CK",
            "fetched_at": datetime.now().isoformat(),
            "techniques": techniques
        }
        
        with open(mitre_output_path, "w") as f:
            json.dump(mitre_output, f, indent=2)
        
        print(f"Saved MITRE ATT&CK data to {mitre_output_path}")
    
    # Print summary
    print("\n" + "=" * 50)
    print("FETCH SUMMARY")
    print("=" * 50)
    print(f"CVEs fetched: {len(cves)}")
    
    if cves:
        severities = {}
        for cve in cves:
            sev = cve.get("severity") or "Unknown"
            severities[sev] = severities.get(sev, 0) + 1
        
        print("\nBy severity:")
        for sev, count in sorted(severities.items()):
            print(f"  {sev}: {count}")
        
        print(f"\nSample CVEs:")
        for cve in cves[:5]:
            print(f"  - {cve['id']}: {cve.get('severity', 'N/A')} ({cve.get('cvss_score', 'N/A')})")
    
    # Auto-ingest if requested
    if args.ingest:
        print("\nIngesting into RAG system...")
        from src.pipeline import ThreatIntelRAG
        from src.data_loader import Document
        
        rag = ThreatIntelRAG(use_reranker=False, check_hallucinations=False)
        
        # Convert to Document objects
        documents = []
        for cve in cves:
            documents.append(Document(
                id=cve["id"],
                title=cve["title"],
                content=cve["content"],
                source="NVD",
                metadata={
                    "type": "cve",
                    "severity": cve.get("severity"),
                    "cvss_score": cve.get("cvss_score"),
                    "published": cve.get("published")
                }
            ))
        
        if args.include_mitre:
            for tech in techniques:
                documents.append(Document(
                    id=tech["id"],
                    title=tech["title"],
                    content=tech["content"],
                    source="MITRE ATT&CK",
                    metadata=tech.get("metadata", {"type": "mitre_attack"})
                ))
        
        rag.ingest_documents(documents)
        print("Ingestion complete!")


if __name__ == "__main__":
    main()
