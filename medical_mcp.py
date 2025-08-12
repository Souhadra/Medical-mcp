import os
import json
import asyncio
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
import httpx
from fastmcp import FastMCP
from fastmcp.server.auth.providers.bearer import BearerAuthProvider, RSAKeyPair
from mcp.server.auth.provider import AccessToken
from mcp.types import TextContent, ImageContent, INVALID_PARAMS, INTERNAL_ERROR
from pydantic import BaseModel, Field, AnyUrl
from bs4 import BeautifulSoup
import random
import time

# --- Load environment variables ---
TOKEN = os.environ.get("PUCH_AUTH_TOKEN")
MY_NUMBER = os.environ.get("MY_NUMBER")

assert TOKEN is not None, "Please set PUCH_AUTH_TOKEN in your .env file"
assert MY_NUMBER is not None, "Please set MY_NUMBER in your .env file"

# --- Constants ---
FDA_API_BASE = "https://api.fda.gov"
WHO_API_BASE = "https://ghoapi.azureedge.net/api"
RXNAV_API_BASE = "https://rxnav.nlm.nih.gov/REST"
PUBMED_API_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
GOOGLE_SCHOLAR_API_BASE = "https://scholar.google.com/scholar"
USER_AGENT = "medical-mcp/1.0"

# --- Auth Provider ---
class SimpleBearerAuthProvider(BearerAuthProvider):
    def __init__(self, token: str):
        k = RSAKeyPair.generate()
        super().__init__(public_key=k.public_key, jwks_uri=None, issuer=None, audience=None)
        self.token = token

    async def load_access_token(self, token: str) -> AccessToken | None:
        if token == self.token:
            return AccessToken(
                token=token,
                client_id="puch-client",
                scopes=["*"],
                expires_at=None,
            )
        return None

# --- Rich Tool Description model ---
class RichToolDescription(BaseModel):
    description: str
    use_when: str
    side_effects: str | None = None

# --- MCP Server Setup ---
mcp = FastMCP(
    "Medical MCP Server",
    auth=SimpleBearerAuthProvider(TOKEN),
)

# --- Data Models ---
class DrugLabel(BaseModel):
    openfda: Dict[str, List[str]]
    purpose: List[str]
    warnings: List[str]
    drug_interactions: List[str]
    effective_time: str

class WHOIndicator(BaseModel):
    SpatialDim: str
    TimeDim: str
    Value: str
    Comments: str
    NumericValue: float
    Low: Optional[float]
    High: Optional[float]
    Date: str

class RxNormDrug(BaseModel):
    name: str
    rxcui: str
    tty: str
    language: str
    synonym: List[str]

class PubMedArticle(BaseModel):
    pmid: str
    title: str
    abstract: str
    authors: List[str]
    journal: str
    publication_date: str

class GoogleScholarArticle(BaseModel):
    title: str
    authors: str
    abstract: str
    journal: str
    year: str
    citations: str
    url: str

# --- Utility Functions ---
async def search_drugs(query: str, limit: int = 10) -> List[DrugLabel]:
    async with httpx.AsyncClient() as client:
        res = await client.get(
            f"{FDA_API_BASE}/drug/label.json",
            params={"search": f"openfda.brand_name:{query}", "limit": limit},
            headers={"User-Agent": USER_AGENT}
        )
        return res.json().get("results", [])

async def get_drug_by_ndc(ndc: str) -> Optional[DrugLabel]:
    async with httpx.AsyncClient() as client:
        res = await client.get(
            f"{FDA_API_BASE}/drug/label.json",
            params={"search": f"openfda.product_ndc:{ndc}", "limit": 1},
            headers={"User-Agent": USER_AGENT}
        )
        results = res.json().get("results", [])
        return results[0] if results else None

async def get_health_indicators(indicator_name: str, country: Optional[str] = None) -> List[WHOIndicator]:
    filter_query = f"IndicatorName eq '{indicator_name}'"
    if country:
        filter_query += f" and SpatialDim eq '{country}'"

    async with httpx.AsyncClient() as client:
        res = await client.get(
            f"{WHO_API_BASE}/Indicator",
            params={"$filter": filter_query, "$format": "json"},
            headers={"User-Agent": USER_AGENT}
        )
        return res.json().get("value", [])

async def search_rxnorm_drugs(query: str) -> List[RxNormDrug]:
    async with httpx.AsyncClient() as client:
        res = await client.get(
            f"{RXNAV_API_BASE}/drugs.json",
            params={"name": query},
            headers={"User-Agent": USER_AGENT}
        )
        return res.json().get("drugGroup", {}).get("conceptGroup", [{}])[0].get("concept", [])

async def random_delay(min_delay: float, max_delay: float):
    delay = random.uniform(min_delay, max_delay)
    await asyncio.sleep(delay)

async def search_google_scholar(query: str) -> List[GoogleScholarArticle]:
    await random_delay(1.0, 3.0)

    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache"
    }

    async with httpx.AsyncClient(headers=headers) as client:
        search_url = f"{GOOGLE_SCHOLAR_API_BASE}?q={query}&hl=en"
        res = await client.get(search_url)
        soup = BeautifulSoup(res.text, 'html.parser')
        results = []

        for element in soup.select(".gs_r, .gs_ri, [data-rp]"):
            title_element = element.select_one(".gs_rt a, .gs_rt, h3 a, h3") or element.select_one("a[data-clk]") or element.select_one("h3")
            title = title_element.get_text(strip=True) if title_element else ""
            url = title_element.get('href') if title_element else ""

            authors_element = element.select_one(".gs_a, .gs_authors, .gs_venue") or element.select_one('[class*="author"]') or element.select_one('[class*="venue"]')
            authors = authors_element.get_text(strip=True) if authors_element else ""

            abstract_element = element.select_one(".gs_rs, .gs_rs_a, .gs_snippet") or element.select_one('[class*="snippet"]') or element.select_one('[class*="abstract"]')
            abstract = abstract_element.get_text(strip=True) if abstract_element else ""

            citations_element = element.select_one(".gs_fl a, .gs_fl") or element.select_one('[class*="citation"]') or element.select_one('a[href*="cites"]')
            citations = citations_element.get_text(strip=True) if citations_element else ""

            year = ""
            year_match = next((m.group(1) for m in [re.search(r"(\d{4})", text) for text in [authors, title, abstract]] if m), None)
            if year_match:
                year = year_match

            journal = ""
            journal_match = next((m.group(1) for m in [re.search(r"- ([^-]+)$", authors), re.search(r", ([^,]+)$", authors), re.search(r"in ([^,]+)", authors)] if m), None)
            if journal_match:
                journal = journal_match.strip()

            if title and len(title) > 5:
                results.append({
                    "title": title,
                    "authors": authors,
                    "abstract": abstract,
                    "journal": journal,
                    "year": year,
                    "citations": citations,
                    "url": url
                })

        return results

async def search_pubmed_articles(query: str, max_results: int = 10) -> List[PubMedArticle]:
    async with httpx.AsyncClient() as client:
        search_res = await client.get(
            f"{PUBMED_API_BASE}/esearch.fcgi",
            params={"db": "pubmed", "term": query, "retmode": "json", "retmax": max_results},
            headers={"User-Agent": USER_AGENT}
        )
        id_list = search_res.json().get("esearchresult", {}).get("idlist", [])

        if not id_list:
            return []

        fetch_res = await client.get(
            f"{PUBMED_API_BASE}/efetch.fcgi",
            params={"db": "pubmed", "id": ",".join(id_list), "retmode": "xml"},
            headers={"User-Agent": USER_AGENT}
        )

        articles = []
        xml_text = fetch_res.text
        pmid_matches = re.findall(r'<PMID[^>]*>(\d+)<\/PMID>', xml_text)
        title_matches = re.findall(r'<ArticleTitle[^>]*>([^<]+)<\/ArticleTitle>', xml_text)

        for pmid, title in zip(pmid_matches, title_matches):
            articles.append({
                "pmid": pmid,
                "title": title,
                "abstract": "Abstract not available in this format",
                "authors": [],
                "journal": "Journal information not available",
                "publication_date": "Date not available"
            })

        return articles

# --- MCP Tools ---
@mcp.tool
async def validate() -> str:
    return MY_NUMBER

# --- Tool: search_drugs ---
SearchDrugsDescription = RichToolDescription(
    description="Search for drug information using FDA database",
    use_when="Use this to search for drug information by brand or generic name.",
    side_effects="Returns a list of drugs matching the search criteria.",
)

@mcp.tool(description=SearchDrugsDescription.model_dump_json())
async def search_drugs_tool(
    query: str = Field(..., description="Drug name to search for (brand name or generic name)"),
    limit: int = Field(10, description="Number of results to return (max 50)", ge=1, le=50)
) -> str:
    try:
        drugs = await search_drugs(query, limit)

        if not drugs:
            return f"No drugs found matching '{query}'. Try a different search term."

        result = f"**Drug Search Results for '{query}'**\n\n"
        result += f"Found {len(drugs)} drug(s)\n\n"

        for index, drug in enumerate(drugs, start=1):
            result += f"{index}. **{drug['openfda'].get('brand_name', ['Unknown Brand'])[0]}**\n"
            result += f"   Generic Name: {drug['openfda'].get('generic_name', ['Not specified'])[0]}\n"
            result += f"   Manufacturer: {drug['openfda'].get('manufacturer_name', ['Not specified'])[0]}\n"
            result += f"   Route: {drug['openfda'].get('route', ['Not specified'])[0]}\n"
            result += f"   Dosage Form: {drug['openfda'].get('dosage_form', ['Not specified'])[0]}\n"

            if drug.get('purpose') and drug['purpose']:
                result += f"   Purpose: {drug['purpose'][0][:200]}{'...' if len(drug['purpose'][0]) > 200 else ''}\n"

            result += f"   Last Updated: {drug['effective_time']}\n\n"

        return result
    except Exception as e:
        return f"Error searching drugs: {str(e)}"

# --- Tool: get_drug_details ---
GetDrugDetailsDescription = RichToolDescription(
    description="Get detailed information about a specific drug by NDC (National Drug Code)",
    use_when="Use this to get detailed information about a specific drug using its NDC.",
    side_effects="Returns detailed information about the specified drug.",
)

@mcp.tool(description=GetDrugDetailsDescription.model_dump_json())
async def get_drug_details_tool(
    ndc: str = Field(..., description="National Drug Code (NDC) of the drug")
) -> str:
    try:
        drug = await get_drug_by_ndc(ndc)

        if not drug:
            return f"No drug found with NDC: {ndc}"

        result = f"**Drug Details for NDC: {ndc}**\n\n"
        result += "**Basic Information:**\n"
        result += f"- Brand Name: {drug['openfda'].get('brand_name', ['Not specified'])[0]}\n"
        result += f"- Generic Name: {drug['openfda'].get('generic_name', ['Not specified'])[0]}\n"
        result += f"- Manufacturer: {drug['openfda'].get('manufacturer_name', ['Not specified'])[0]}\n"
        result += f"- Route: {drug['openfda'].get('route', ['Not specified'])[0]}\n"
        result += f"- Dosage Form: {drug['openfda'].get('dosage_form', ['Not specified'])[0]}\n"
        result += f"- Last Updated: {drug['effective_time']}\n\n"

        if drug.get('purpose') and drug['purpose']:
            result += "**Purpose/Uses:**\n"
            for index, purpose in enumerate(drug['purpose'], start=1):
                result += f"{index}. {purpose}\n"
            result += "\n"

        if drug.get('warnings') and drug['warnings']:
            result += "**Warnings:**\n"
            for index, warning in enumerate(drug['warnings'], start=1):
                result += f"{index}. {warning[:300]}{'...' if len(warning) > 300 else ''}\n"
            result += "\n"

        if drug.get('drug_interactions') and drug['drug_interactions']:
            result += "**Drug Interactions:**\n"
            for index, interaction in enumerate(drug['drug_interactions'], start=1):
                result += f"{index}. {interaction[:300]}{'...' if len(interaction) > 300 else ''}\n"
            result += "\n"

        return result
    except Exception as e:
        return f"Error fetching drug details: {str(e)}"

# --- Tool: get_health_statistics ---
GetHealthStatisticsDescription = RichToolDescription(
    description="Get health statistics and indicators from WHO Global Health Observatory",
    use_when="Use this to get health statistics and indicators from WHO.",
    side_effects="Returns health statistics matching the search criteria.",
)

@mcp.tool(description=GetHealthStatisticsDescription.model_dump_json())
async def get_health_statistics_tool(
    indicator: str = Field(..., description="Health indicator to search for (e.g., 'Life expectancy', 'Mortality rate')"),
    country: Optional[str] = Field(None, description="Country code (e.g., 'USA', 'GBR') - optional"),
    limit: int = Field(10, description="Number of results to return (max 20)", ge=1, le=20)
) -> str:
    try:
        indicators = await get_health_indicators(indicator, country)

        if not indicators:
            return f"No health indicators found for '{indicator}'{' in ' + country if country else ''}. Try a different search term."

        result = f"**Health Statistics: {indicator}**\n\n"
        if country:
            result += f"Country: {country}\n"
        result += f"Found {len(indicators)} data points\n\n"

        for index, ind in enumerate(indicators[:limit], start=1):
            result += f"{index}. **{ind['SpatialDim']}** ({ind['TimeDim']})\n"
            result += f"   Value: {ind['Value']} {ind['Comments'] or ''}\n"
            result += f"   Numeric Value: {ind['NumericValue']}\n"
            if ind.get('Low') and ind.get('High'):
                result += f"   Range: {ind['Low']} - {ind['High']}\n"
            result += f"   Date: {ind['Date']}\n\n"

        return result
    except Exception as e:
        return f"Error fetching health statistics: {str(e)}"

# --- Tool: search_medical_literature ---
SearchMedicalLiteratureDescription = RichToolDescription(
    description="Search for medical research articles in PubMed",
    use_when="Use this to search for medical research articles in PubMed.",
    side_effects="Returns a list of medical research articles matching the search criteria.",
)

@mcp.tool(description=SearchMedicalLiteratureDescription.model_dump_json())
async def search_medical_literature_tool(
    query: str = Field(..., description="Medical topic or condition to search for"),
    max_results: int = Field(10, description="Maximum number of articles to return (max 20)", ge=1, le=20)
) -> str:
    try:
        articles = await search_pubmed_articles(query, max_results)

        if not articles:
            return f"No medical articles found for '{query}'. Try a different search term."

        result = f"**Medical Literature Search: '{query}'**\n\n"
        result += f"Found {len(articles)} article(s)\n\n"

        for index, article in enumerate(articles, start=1):
            result += f"{index}. **{article['title']}**\n"
            result += f"   PMID: {article['pmid']}\n"
            result += f"   Journal: {article['journal']}\n"
            result += f"   Publication Date: {article['publication_date']}\n"
            if article.get('doi'):
                result += f"   DOI: {article['doi']}\n"
            result += "\n"

        return result
    except Exception as e:
        return f"Error searching medical literature: {str(e)}"

# --- Tool: search_drug_nomenclature ---
SearchDrugNomenclatureDescription = RichToolDescription(
    description="Search for drug information using RxNorm (standardized drug nomenclature)",
    use_when="Use this to search for drug information using RxNorm.",
    side_effects="Returns a list of drugs matching the search criteria in RxNorm.",
)

@mcp.tool(description=SearchDrugNomenclatureDescription.model_dump_json())
async def search_drug_nomenclature_tool(
    query: str = Field(..., description="Drug name to search for in RxNorm database")
) -> str:
    try:
        drugs = await search_rxnorm_drugs(query)

        if not drugs:
            return f"No drugs found in RxNorm database for '{query}'. Try a different search term."

        result = f"**RxNorm Drug Search: '{query}'**\n\n"
        result += f"Found {len(drugs)} drug(s)\n\n"

        for index, drug in enumerate(drugs, start=1):
            result += f"{index}. **{drug['name']}**\n"
            result += f"   RxCUI: {drug['rxcui']}\n"
            result += f"   Term Type: {drug['tty']}\n"
            result += f"   Language: {drug['language']}\n"
            if drug.get('synonym') and drug['synonym']:
                result += f"   Synonyms: {', '.join(drug['synonym'][:3])}{'...' if len(drug['synonym']) > 3 else ''}\n"
            result += "\n"

        return result
    except Exception as e:
        return f"Error searching RxNorm: {str(e)}"

# --- Tool: search_google_scholar ---
SearchGoogleScholarDescription = RichToolDescription(
    description="Search for academic research articles using Google Scholar",
    use_when="Use this to search for academic research articles using Google Scholar.",
    side_effects="Returns a list of academic research articles matching the search criteria.",
)

@mcp.tool(description=SearchGoogleScholarDescription.model_dump_json())
async def search_google_scholar_tool(
    query: str = Field(..., description="Academic topic or research query to search for")
) -> str:
    try:
        articles = await search_google_scholar(query)

        if not articles:
            return f"No academic articles found for '{query}'. This could be due to:\n- No results matching your query\n- Google Scholar rate limiting\n- Network connectivity issues\n\nTry refining your search terms or try again later."

        result = f"**Google Scholar Search: '{query}'**\n\n"
        result += f"Found {len(articles)} article(s)\n\n"

        for index, article in enumerate(articles, start=1):
            result += f"{index}. **{article['title']}**\n"
            if article.get('authors'):
                result += f"   Authors: {article['authors']}\n"
            if article.get('journal'):
                result += f"   Journal: {article['journal']}\n"
            if article.get('year'):
                result += f"   Year: {article['year']}\n"
            if article.get('citations'):
                result += f"   Citations: {article['citations']}\n"
            if article.get('url'):
                result += f"   URL: {article['url']}\n"
            if article.get('abstract'):
                result += f"   Abstract: {article['abstract'][:300]}{'...' if len(article['abstract']) > 300 else ''}\n"
            result += "\n"

        return result
    except Exception as e:
        return f"Error searching Google Scholar: {str(e)}. This might be due to rate limiting or network issues. Please try again later."

# --- Run MCP Server ---
async def main():
    print("🚀 Starting Medical MCP server on http://0.0.0.0:8086")
    await mcp.run_async("streamable-http", host="0.0.0.0", port=8086)

if __name__ == "__main__":
    import re
    asyncio.run(main())
