"""Summarisation tool for generating audience-specific summaries."""

from datetime import date
from typing import Dict, List, Literal, Optional

from config.settings import settings
from clients.summariser_client import SummariserClient
from clients.summary_client import SummaryClient
from models.llm_inputs import SummariserInput, SummaryInput
from models.llm_outputs import (
    HypothesisResult,
    InsightsEngineOutput,
    SummariserOutput,
    SummaryOutput,
)


class Summariser:
    """
    Unified summarisation tool combining general summarisation and analysis summaries.

    Provides:
    - General content summarisation with audience/tone adaptation
    - Analysis summary generation for hypotheses and insights
    """

    def __init__(self, use_llm: bool = True):
        """
        Initialize the summariser.

        Args:
            use_llm: Whether to use LLM for summarisation
        """
        self.use_llm = use_llm

        if use_llm:
            self.summariser_client = SummariserClient()
            self.summary_client = SummaryClient()
        else:
            self.summariser_client = None
            self.summary_client = None

    def summarise(
        self,
        content: str,
        audience_goal: str,
        expertise: Literal["low", "medium", "high"] = "medium",
        tone: Literal["formal", "conversational", "urgent", "encouraging"] = "formal",
        format_type: Literal["paragraph", "bullets", "structured"] = "structured",
        key_points_limit: int = 5
    ) -> SummariserOutput:
        """
        Summarise content for a specific audience.

        Args:
            content: Content to summarise
            audience_goal: What the audience needs from the summary
            expertise: Audience expertise level
            tone: Desired tone
            format_type: Output format preference
            key_points_limit: Maximum key points

        Returns:
            SummariserOutput with summary and key points
        """
        if self.use_llm and self.summariser_client:
            input_data = SummariserInput(
                content=content,
                audience_goal=audience_goal,
                audience_expertise=expertise,
                tone=tone,
                format=format_type,
                key_points_limit=key_points_limit
            )
            return self.summariser_client.summarise(input_data)
        else:
            return self._summarise_rule_based(content, key_points_limit, tone)

    def _summarise_rule_based(
        self,
        content: str,
        key_points_limit: int,
        tone: str
    ) -> SummariserOutput:
        """Simple rule-based summarisation."""
        # Split into sentences
        sentences = content.replace("\n", " ").split(". ")

        # Take first few sentences as summary
        summary_sentences = sentences[:min(3, len(sentences))]
        summary = ". ".join(summary_sentences)
        if not summary.endswith("."):
            summary += "."

        # Extract key points (first sentence of each paragraph)
        paragraphs = content.split("\n\n")
        key_points = []
        for para in paragraphs[:key_points_limit]:
            first_sentence = para.split(". ")[0]
            if first_sentence and len(first_sentence) > 10:
                key_points.append(first_sentence)

        return SummariserOutput(
            summary=summary,
            key_points=key_points[:key_points_limit],
            tone_applied=tone,
            word_count=len(summary.split()),
            audience_appropriate=True
        )

    def generate_analysis_summary(
        self,
        hypotheses: List[HypothesisResult],
        investigations: List[InsightsEngineOutput],
        audience: Literal["executive", "analyst", "operations"] = "executive",
        tone: Literal["formal", "conversational", "urgent"] = "formal"
    ) -> SummaryOutput:
        """
        Generate a summary of analysis results.

        Args:
            hypotheses: Detected hypotheses
            investigations: Completed investigations
            audience: Target audience
            tone: Desired tone

        Returns:
            SummaryOutput with structured summary
        """
        if self.use_llm and self.summary_client:
            input_data = SummaryInput(
                insights=[
                    {
                        "hypothesis": inv.hypothesis_description,
                        "root_cause": inv.root_cause_summary,
                        "confidence": inv.confidence_score,
                        "recommendations": inv.overall_recommendations
                    }
                    for inv in investigations
                ],
                hypotheses=[
                    {
                        "description": h.description,
                        "significance": h.significance_score,
                        "metric": h.metric
                    }
                    for h in hypotheses
                ],
                audience=audience,
                tone=tone
            )
            return self.summary_client.generate_summary(input_data)
        else:
            return self._generate_summary_rule_based(
                hypotheses, investigations, audience, tone
            )

    def _generate_summary_rule_based(
        self,
        hypotheses: List[HypothesisResult],
        investigations: List[InsightsEngineOutput],
        audience: str,
        tone: str
    ) -> SummaryOutput:
        """Generate summary using rule-based approach."""
        from models.llm_outputs import ActionItem, KeyFinding

        # Generate executive summary
        high_priority = [h for h in hypotheses if h.significance_score > 0.7]

        if high_priority:
            exec_summary = (
                f"Analysis identified {len(hypotheses)} significant changes, "
                f"with {len(high_priority)} requiring immediate attention. "
            )
            if investigations:
                top_cause = max(investigations, key=lambda i: i.confidence_score)
                exec_summary += f"Primary root cause: {top_cause.root_cause_summary}"
        else:
            exec_summary = (
                f"Analysis reviewed {len(hypotheses)} potential issues. "
                "No critical items require immediate action."
            )

        # Generate key findings
        key_findings = []
        for i, hyp in enumerate(hypotheses[:5]):
            finding = KeyFinding(
                title=f"{hyp.metric.replace('_', ' ').title()} Change",
                description=hyp.description,
                impact="high" if hyp.significance_score > 0.7 else "medium",
                affected_areas=[hyp.hierarchy_name]
            )
            key_findings.append(finding)

        # Generate action items
        action_items = []
        for inv in investigations:
            for rec in inv.overall_recommendations[:2]:
                item = ActionItem(
                    action=rec,
                    priority="immediate" if inv.confidence_score > 0.7 else "short_term",
                    expected_impact="Address identified root cause"
                )
                action_items.append(item)

        return SummaryOutput(
            audience=audience,
            tone=tone,
            generated_date=date.today(),
            executive_summary=exec_summary,
            key_findings=key_findings,
            action_items=action_items[:5]
        )

    def summarise_for_email(
        self,
        content: str,
        recipient_role: str = "manager"
    ) -> str:
        """
        Create an email-appropriate summary.

        Args:
            content: Content to summarise
            recipient_role: Role of email recipient

        Returns:
            Email-ready summary text
        """
        expertise = "high" if recipient_role in ["analyst", "data scientist"] else "medium"
        tone = "formal" if recipient_role in ["executive", "director"] else "conversational"

        result = self.summarise(
            content=content,
            audience_goal=f"Brief {recipient_role} on key findings",
            expertise=expertise,
            tone=tone,
            format_type="structured"
        )

        # Format for email
        email_text = f"{result.summary}\n\nKey Points:\n"
        for point in result.key_points:
            email_text += f"• {point}\n"

        return email_text

    def generate_alert(
        self,
        hypotheses: List[HypothesisResult],
        investigations: List[InsightsEngineOutput]
    ) -> Optional[Dict]:
        """
        Generate an alert if critical issues are found.

        Args:
            hypotheses: Detected hypotheses
            investigations: Completed investigations

        Returns:
            Alert dictionary if critical issues exist, None otherwise
        """
        critical = [h for h in hypotheses if h.significance_score > 0.8]

        if not critical:
            return None

        if self.use_llm and self.summary_client:
            issues = [
                {"description": h.description, "significance": h.significance_score}
                for h in critical
            ]
            impact = {
                "affected_hypotheses": len(critical),
                "investigations_completed": len(investigations)
            }
            return self.summary_client.generate_alert(issues, impact)
        else:
            # Rule-based alert
            return {
                "severity": "critical" if any(h.significance_score > 0.9 for h in critical) else "high",
                "headline": f"{len(critical)} critical issues require immediate attention",
                "situation": critical[0].description if critical else "Multiple issues detected",
                "immediate_actions": ["Review flagged metrics", "Investigate root causes"],
                "escalation": ["Operations Manager", "Regional Lead"]
            }

    def extract_key_metrics(
        self,
        investigations: List[InsightsEngineOutput]
    ) -> Dict:
        """
        Extract key metrics from investigations for dashboards.

        Args:
            investigations: Completed investigations

        Returns:
            Dictionary of key metrics
        """
        metrics = {
            "total_hypotheses_investigated": len(investigations),
            "high_confidence_findings": 0,
            "root_causes_identified": [],
            "affected_levels": set(),
            "top_recommendations": []
        }

        for inv in investigations:
            if inv.confidence_score > 0.7:
                metrics["high_confidence_findings"] += 1
                metrics["root_causes_identified"].append({
                    "hypothesis": inv.hypothesis_description,
                    "root_cause": inv.root_cause_summary,
                    "confidence": inv.confidence_score
                })

            for insight in inv.insights:
                metrics["affected_levels"].add(insight.hierarchy_level)

            metrics["top_recommendations"].extend(inv.overall_recommendations)

        # Convert set to list for JSON serialization
        metrics["affected_levels"] = list(metrics["affected_levels"])
        # Deduplicate recommendations
        metrics["top_recommendations"] = list(dict.fromkeys(metrics["top_recommendations"]))[:10]

        return metrics
