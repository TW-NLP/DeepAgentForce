#!/usr/bin/env python3
"""
Search the web using Tavily API
Part of the web-search Agent Skill
"""

import argparse
import sys
import json
import os
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))
from src.services.base import BaseConfigurableService

class TAVILYSearch(BaseConfigurableService):
    """Tavily Web Search Skill"""
    
    def __init__(self):
        super().__init__()
    def build_instance():
        pass
    

    def web_search(self,query, max_results=5, output=None):
        """Execute web search"""
        try:
            from tavily import TavilyClient
        except ImportError as e:
            print(f"❌ Error: Missing dependency: {e}")
            print("   Install with: pip install tavily-python")
            return 1
        
        try:
            
            # Initialize client
            client = TavilyClient(api_key=self.settings.TAVILY_API_KEY)
            
            print(f"🔍 Searching for: {query}")
            
            # Execute search
            response = client.search(query=query, max_results=max_results)
            
            results = response.get('results', [])
            
            if not results:
                print("ℹ️  No results found")
                return 1
            
            # Prepare structured output for LLM
            structured_output = {
                "query": query,
                "total_results": len(results),
                "results": [
                    {
                        "title": result.get('title', 'No title'),
                        "url": result.get('url', 'N/A'),
                        "snippet": result.get('content', 'No description')[:500]
                    }
                    for result in results
                ]
            }
            
            # Save to file if output specified
            if output:
                output_file = Path(output)
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(response, f, ensure_ascii=False, indent=2)
            
            # ALWAYS output JSON to stdout for LLM to parse
            print(json.dumps(structured_output, ensure_ascii=False, indent=2))
            
            return 0
        
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
            return 1

def main():
    tavily_search = TAVILYSearch()
    parser = argparse.ArgumentParser(
        description='Search the web using Tavily API',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic search
  %(prog)s "Python programming tutorials"
  
  # Limit results
  %(prog)s "AI news" --max-results 10
  
  # Save to file
  %(prog)s "machine learning" --output results.json
        """
    )
    
    parser.add_argument('query', help='Search query')
    parser.add_argument('--max-results', type=int, default=5,
                       help='Maximum number of results (default: 5)')
    parser.add_argument('--output', help='Output file path for JSON results')
    
    args = parser.parse_args()
    
    sys.exit(tavily_search.web_search(args.query, args.max_results, args.output))

if __name__ == '__main__':
    main()