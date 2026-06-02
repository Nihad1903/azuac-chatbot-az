import os
import logging
import nest_asyncio
import re
import csv
import asyncio
import pandas as pd
import fitz
from datetime import datetime
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
nest_asyncio.apply()
load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
UNIVERSITY_PDF  = os.getenv("UNIVERSITY_PDF", "Azmiu_info.pdf")
BOT_NAME        = "ScholaraBot"
UNIVERSITY_NAME = "Azərbaycan Memarlıq və İnşaat Universiteti (AzMİU)"
LOG_FILE        = "version1_azmiu_scholara_logs.csv"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# ── Abbreviatura lüğəti ───────────────────────────────────────────────────────
ABBREVIATION_MAP = {
    "it":   ["informasiya texnologiyaları", "informasiya təhlükəsizliyi", "informasiya texnologiyaları mühəndisliyi"],
    "iт":   ["informasiya texnologiyaları", "informasiya təhlükəsizliyi"],
    "cs":   ["informasiya texnologiyaları", "kompüter elmləri"],
    "ai":   ["süni intellekt", "informasiya texnologiyaları"],
    "ii":   ["inşaat mühəndisliyi"],
    "nm":   ["nəqliyyat mühəndisliyi"],
    "em":   ["ekologiya mühəndisliyi"],
    "ee":   ["elektrik elektronika mühəndisliyi"],
    "dim":  ["dövlət imtahan mərkəzi"],
    "azmiu":["azərbaycan memarlıq inşaat universiteti"],
    "tgt":  ["tələbə gənclər təşkilatı"],
    "thik": ["tələbə həmkarlar ittifaqı komitəsi"],
    "etş":  ["elmi tədqiqat şöbəsi"],
    "ikt":  ["informasiya kommunikasiya texnologiyaları"],
    "gpa":  ["akademik göstərici orta bal"],
    "üomg": ["ümumiləşdirilmiş orta məktəb göstəricisi"],
    "polimi": ["milan texniki universiteti"],
    "itu":  ["istanbul texniki universiteti"],
    "itü":  ["istanbul texniki universiteti"],
}

SECTION_HEADER_PATTERNS = [
    (r"^(Memarlıq|İnşaat|Su təsərrüfatı|Nəqliyyat|Dizayn|Tikinti-iqtisad|Mexanika|Əcnəbi|İxtisasartırma)\s+fakültəsi", "fakülte"),
    (r"^\d+.\s+(Memarlıq|Şəhərsalma|İnşaat|Materiallar|Sənaye|Kommunikasiya|Meliorasiya|Ekologiya|Neft-qaz|Elektrik|Yer|Geomatika|Logistika|Nəqliyyat|Həyat|Dizayn|İqtisadiyyat|Menecment|Marketinq|Biznesin|İnformasiya|Maşın|Mexanika|Proseslərin)", "ixtisas"),
    (r"(İNŞAAT KOLLECİ|İnşaat Kolleci|Kollecinin|kollec)", "kollec_bölməsi"),
    (r"(Mərkəz haqqında|mərkəzi\s*$)", "merkez"),
    (r"(Şöbə haqqında|şöbəsi\s*$)", "sobe"),
    (r"(İKİLİ DİPLOM|MAGİSTRATURA PROQRAM|BAKALAVR PROQRAM)", "diplom"),
    (r"(Qəbul Balları|qəbul balı|AzMİU.Qəbul)", "qebul_bal"),
    (r"(Elmi [Şş]ura|Elmi.[Şş]uranın)", "elmi_sura"),
    (r"SABAH", "sabah"),
]

BASE_SYSTEM_PROMPT = """Sən {BOT_NAME} robotusan, {UNIVERSITY_NAME} üzrə rəsmi tələbə məlumatlandırma sistemisən.
Sənin fəaliyyətin rəsmi sənədə əsaslanır və heç bir halda qaydalardan kənara çıxa bilməzsən.

STRİKT CAVABLANDIRMA QAYDALARI:
1. İstifadəçi hər hansı siyahı (məsələn: fakültələr, ikili diplom imkanları, mövcud klublar/təşkilatlar, təqaüdlər və ya keçid balları) tələb etdikdə, təqdim olunan rəsmi məlumatda həmin mövzuya aid olan BÜTÜN bəndləri və ixtisasları İSTİSNASIZ olaraq alt-alta tam çıxarmalısan. 1-2 nümunə göstərib dayanmaq, məlumatı qısaltmaq və ya ümumiləşdirmək qətiyyən QADAĞANDIR. 
2. Fakültə haqqında sual veriləndə yalnız fakültənin strukturu, rəhbərliyi və ümumi göstəricilərini təqdim et. Başqa bölmələrdəki ixtisas kataloqlarının daxili geniş təsvirlərini bura qarışdırma.
3. Rəsmi mətndə yazılmayan heç bir faktı, şərti, mərhələni və ya proqramı özündən əlavə etmə (Məsələn, SABAH qrupları haqqında mətndə nə yazılıbsa, nöqtə-vergülünə qədər yalnız onu istifadə et).
4. Cavabların yalnız rəsmi Azərbaycan ədəbi dilində olmalıdır. Türkiyə türkcəsinə aid olan sözlər, şəkilçilər və ya ifadələr (məsələn: "Elbette", "Hangi", "Umarım", "Yapar", "Kılar") işlədilə bilməz.
5. İstifadəçi öz balını qeyd edib şansını soruşduqda (məsələn: "350 balım var"), gələn rəsmi keçid balları siyahısındakı BÜTÜN ixtisasları yoxla. Balı rəsmi keçid balına çatan və ya ondan yüksək olan BÜTÜN ixtisasları siyahı halında təqdim et. Balının çatmadığı ixtisasları isə dəqiq formada "balınız çatmır" deyərək bildir.
6. "Sənədə əsasən", "PDF-də qeyd olunub", "Mən bir dil modeliyəm" kimi süni ifadələr işlətmə. Universitetin rəsmi nümayəndəsi kimi birbaşa və net danış.
7. Əgər sənəddə sualın cavabı birbaşa yoxdursa, özündən heç nə uydurma və rəsmi şəkildə "Bu barədə rəsmi məlumatımız yoxdur." de."""

class UniBotRAG:
    def __init__(self):
        logging.info("UniBotRAG GPT-4o-mini və Page-Level axtarışla başladılır...")
        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY is not set.")
        self.embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")
        self.reranker = CrossEncoder("BAAI/bge-reranker-large")
        
        # Sürət, dəqiqlik və büdcə baxımından ən optimal model
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.0,
            max_tokens=2548,
            openai_api_key=OPENAI_API_KEY
        )
        
        self.vector_store = None
        self.parent_docs: dict[int, str] = {}
        self.page_section_types: dict[int, set[str]] = {}

    @staticmethod
    def _detect_section_type(text: str) -> set[str]:
        found = set()
        for pattern, stype in SECTION_HEADER_PATTERNS:
            if re.search(pattern, text, re.MULTILINE | re.IGNORECASE):
                found.add(stype)
        return found if found else {"diger"}

    def load_university_pdf(self, file_path: str) -> bool:
        try:
            doc = fitz.open(file_path)
            pages_documents: list[Document] = []
            
            for page_idx, page in enumerate(doc):
                display_page = page_idx + 1
                page_text = page.get_text("text", flags=fitz.TEXT_PRESERVE_LIGATURES | fitz.TEXT_PRESERVE_WHITESPACE)
                clean_page = re.sub(r"[ \t]+", " ", page_text).strip()
                if not clean_page:
                    continue
                
                # Səhifə bütövlüyünü qoruyuruq - Parçalama yoxdur!
                self.parent_docs[display_page] = clean_page
                stypes = self._detect_section_type(clean_page)
                self.page_section_types[display_page] = stypes
                
                # E5 modeli üçün 'passage: ' prefiksi vacibdir
                pages_documents.append(
                    Document(
                        page_content=f"passage: {clean_page}",
                        metadata={
                            "page": display_page,
                            "raw_chunk": clean_page,
                            "section_types": list(stypes)
                        }
                    )
                )
                
            if not pages_documents:
                return False
                
            self.vector_store = QdrantVectorStore.from_documents(
                pages_documents, self.embeddings,
                path="./qdrant_db", 
                collection_name="unibot_core"
            )
            logging.info(f"Bilik bazası bütöv səhifə olaraq yükləndi. Səhifə sayısı: {len(self.parent_docs)}")
            return True
        except Exception as e:
            logging.error(f"PDF yükləmə xətası: {e}")
            return False

    @staticmethod
    def _expand_abbreviations(question: str) -> str:
        q_lower = question.lower()
        expansions = []
        for abbr, full_list in ABBREVIATION_MAP.items():
            pattern = r"(?<![a-zəıöüğşç])" + re.escape(abbr) + r"(?![a-zəıöüğşç])"
            if re.search(pattern, q_lower):
                expansions.extend(full_list)
        if expansions:
            return question + " " + " ".join(expansions)
        return question

    @staticmethod
    def _classify_query(question: str) -> dict:
        q = question.lower()
        
        is_dual  = any(kw in q for kw in ["ikili diplom", "2li diplom", "dual", "double degree", "milan", "istanbul"])
        is_bal   = any(kw in q for kw in ["bal", "qəbul", "keçid balı", "yığmışam", "toplamışam", "şansım"])
        is_sura  = any(kw in q for kw in ["elmi şura", "şura"])
        is_sabah = any(kw in q for kw in ["sabah qrupu", "sabah qrupları", "sabah-a qəbul", "sabah mərkəzi"])
        is_fklt  = any(kw in q for kw in ["fakültə", "fakültəsi"])
        is_club  = any(kw in q for kw in ["klub", "təşkilat", "tgt", "thik", "ictimai", "sosial"])
        is_kollec = any(kw in q for kw in ["kollec", "inşaat kolleci", "tabeliyində", "kollecdə", "kollecin"])
        
        target_sections: set[str] = set()
        if is_kollec:
            target_sections.add("kollec_bölməsi")
            return {
                "max_pages": 6, 
                "k": 25, 
                "expand": True, 
                "target_sections": target_sections, 
                "exclude_sections": {"ixtisas", "qebul_bal", "fakülte"}
            }
        if is_dual:
            target_sections.add("diplom")
            return {"max_pages": 10, "k": 20, "expand": True, "target_sections": target_sections}
        if is_bal:
            target_sections.add("qebul_bal")
            return {"max_pages": 12, "k": 25, "expand": True, "target_sections": target_sections}
        if is_sura:
            target_sections.add("elmi_sura")
            return {"max_pages": 5, "k": 10, "expand": False, "target_sections": target_sections}
        if is_sabah:
            target_sections.add("sabah")
            return {"max_pages": 6, "k": 10, "expand": False, "target_sections": target_sections}
        if is_fklt:
            target_sections.add("fakülte")
            return {"max_pages": 6, "k": 15, "expand": True, "target_sections": target_sections, "exclude_sections": {"ixtisas"}}
        if is_club:
            return {"max_pages": 8, "k": 15, "expand": True, "target_sections": set()}
            
        return {"max_pages": 5, "k": 12, "expand": False, "target_sections": set()}

    @staticmethod
    def _expand_query(question: str) -> list[str]:
        q = question.lower()
        queries = [f"query: {question}"]
        
        if any(kw in q for kw in ["ikili", "2li", "dual", "double"]):
            queries.append("query: İkili diplom proqramları magistr və bakalavr xarici universitetlər siyahısı")
        if any(kw in q for kw in ["bal", "qəbul", "keçid"]):
            queries.append("query: AzMİU ixtisasların qəbul balları minimum və maksimum dövlət sifarişi ödəməli")
            queries.append("query: Qrup 1 Qrup 2 ixtisas qəbul balları siyahısı")
        if "sabah" in q:
            queries.append("query: SABAH qrupları üstünlükləri və qəbul şərtləri mərhələləri")
        if "kollec" in q:
            queries.append("query: İNŞAAT KOLLECİ Yaranma Tarixi Fəaliyyət Təhsil Şərtləri Statistik Göstəricilər İxtisasların Siyahısı")    
        return queries

    def _score_pages(self, ranked: list, target_sections: set[str], exclude_sections: set[str], max_pages: int) -> list[tuple[int, str]]:
        page_scores: dict[int, float] = {}
        for score, doc in ranked:
            page_num = doc.metadata.get("page")
            if not page_num: continue
            
            page_stypes = self.page_section_types.get(page_num, {"diger"})
            if exclude_sections and (page_stypes & exclude_sections) and not (page_stypes & target_sections):
                continue
                
            if page_num not in page_scores:
                page_scores[page_num] = score
            else:
                page_scores[page_num] = max(page_scores[page_num], score)
                
            if target_sections and (page_stypes & target_sections):
                page_scores[page_num] += 3.0 # Hədəf bölməyə güclü bonus
                
        top_pages = sorted(page_scores.items(), key=lambda x: x[1], reverse=True)
        selected = [p for p, _ in top_pages[:max_pages]]
        selected.sort() # Səhifələrin sənəd ardıcıllığını qoruyuruq
        return [(p, self.parent_docs[p]) for p in selected]

    def answer(self, question: str) -> tuple[str, list[str]]:
        if self.vector_store is None:
            return "⚠️ Bilik bazası hələ yüklənməyib.", []
            
        params = self._classify_query(question)
        enriched_question = self._expand_abbreviations(question)
        
        search_queries = self._expand_query(enriched_question) if params["expand"] else [f"query: {enriched_question}"]
        
        all_results: list[Document] = []
        seen_pages = set()
        
        for q in search_queries:
            for doc in self.vector_store.similarity_search(q, k=params["k"]):
                page_num = doc.metadata.get("page")
                if page_num not in seen_pages:
                    all_results.append(doc)
                    
        if not all_results:
            return "Bu barədə rəsmi məlumatımız yoxdur.", []
            
        # Reranking mərhələsi - tam orijinal sual ilə
        pairs = [[question, doc.metadata.get("raw_chunk", doc.page_content)] for doc in all_results]
        scores = self.reranker.predict(pairs)
        ranked = sorted(zip(scores, all_results), key=lambda x: x[0], reverse=True)
        
        retrieved_pages = self._score_pages(
            ranked=ranked,
            target_sections=params.get("target_sections", set()),
            exclude_sections=params.get("exclude_sections", set()),
            max_pages=params["max_pages"]
        )
        
        if not retrieved_pages:
            retrieved_pages = self._score_pages(ranked=ranked, target_sections=set(), exclude_sections=set(), max_pages=params["max_pages"])
            
        context_text = "\n\n---\n\n".join(f"[Səhifə {p}]\n{t}" for p, t in retrieved_pages)
        
        current_prompt = BASE_SYSTEM_PROMPT.format(BOT_NAME=BOT_NAME, UNIVERSITY_NAME=UNIVERSITY_NAME)
        
        messages = [
            SystemMessage(content=current_prompt),
            HumanMessage(content=f"RƏSMİ UNİVERSİTET MƏLUMATI:\n{context_text}\n\nTƏLƏBƏNİN SUALI: {question}")
        ]
        
        response = self.llm.invoke(messages).content.strip()
        
        # Robotik hallüsinasiya prefikslərinin təmizlənməsi
        CLEANUP_PATTERNS = [
            r"(?i)based on the (official |provided |university )?information,?\s*",
            r"(?i)according to the (official |university )?data,?\s*",
            r"(?i)kontekstə əsasən,?\s*",
            r"(?i)məlumatlara görə,?\s*",
            r"(?i)təqdim olunan məlumata əsasən,?\s*",
            r"(?i)sənədə əsasən,?\s*",
            r"(?i)pdf-də qeyd olunub,?\s*",
            r"(?i)mən bir dil modeli(yəm)?,?\s*",
        ]
        for pattern in CLEANUP_PATTERNS:
            response = re.sub(pattern, "", response)
            
        raw_chunks_logged = [doc.page_content[:500] for _, doc in ranked[:5]]
        return response.strip(), raw_chunks_logged

# ── Bot instansiyası ─────────────────────────────────────────────────────────
_bot_instance = None

def get_bot() -> "UniBotRAG":
    global _bot_instance
    if _bot_instance is None:
        _bot_instance = UniBotRAG()
        if not _bot_instance.load_university_pdf(UNIVERSITY_PDF):
            raise RuntimeError(f"Failed to load PDF: {UNIVERSITY_PDF}")
    return _bot_instance

# ── Köməkçi funksiyalar ───────────────────────────────────────────────────────
def smart_escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("< ", "&lt; ").replace(" >", " &gt;")

def sanitize_text(text: str) -> str:
    if not text: return ""
    return " ".join(text.replace("\n", " ").replace("\r", " ").split())

def split_message_text(text: str, max_length: int = 1800) -> list[str]:
    if len(text) <= max_length:
        return [text]
    paragraphs = text.split("\n\n")
    chunks, current = [], ""
    for para in paragraphs:
        if len(current) + len(para) + 2 <= max_length:
            current += para + "\n\n"
        else:
            if current: chunks.append(current.strip())
            current = para + "\n\n"
    if current: chunks.append(current.strip())
    return chunks

def log_to_csv(user_id, question, answer, chunks: list[str], status="Baxılmayıb"):
    try:
        data = {
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "User_ID":   user_id,
            "Sual":      sanitize_text(question),
            "Cavab":     sanitize_text(answer),
            "Chunk_1":   sanitize_text(chunks[0]) if len(chunks) > 0 else "",
            "Chunk_2":   sanitize_text(chunks[1]) if len(chunks) > 1 else "",
            "Chunk_3":   sanitize_text(chunks[2]) if len(chunks) > 2 else "",
            "Chunk_4":   sanitize_text(chunks[3]) if len(chunks) > 3 else "",
            "Chunk_5":   sanitize_text(chunks[4]) if len(chunks) > 4 else "",
            "Feedback":  status,
        }
        pd.DataFrame([data]).to_csv(
            LOG_FILE, mode="a", index=False,
            header=not os.path.exists(LOG_FILE),
            encoding="utf-8-sig", quoting=csv.QUOTE_ALL, escapechar="\\"
        )
    except Exception as e:
        logging.error(f"CSV Log xətası: {e}")
# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        get_bot()
        logging.info("RAG initialized successfully.")
    except Exception as exc:
        logging.critical("Initialization failed: %s", exc)
        raise SystemExit(1)
