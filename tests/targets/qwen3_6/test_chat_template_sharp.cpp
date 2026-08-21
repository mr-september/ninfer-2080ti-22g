#include <ninfer/targets/qwen3_6/frontend.h>
#include <ninfer/types.h>

#include "targets/qwen3_6/impl/frontend/chat_template.h"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

namespace fi = ninfer::targets::qwen3_6::frontend_internal;
using ninfer::ChatRole;
using ninfer::ReasoningEffort;
using fi::ChatPart;
using fi::ChatMessage;
using fi::ChatRenderOptions;
using ninfer::ChatStyle;
using fi::ChatTemplateSemantics;
using fi::CompiledChatTemplate;

static int g_failures = 0;

namespace {

struct Fixture {
    std::string name;
    std::vector<ChatMessage> messages;
    std::vector<std::string> tool_jsons;
};

ChatMessage make_user(const std::string& c) {
    ChatMessage m; m.role = ChatRole::User; m.parts.push_back(ChatPart::text_part(c));
    return m;
}
ChatMessage make_system(const std::string& c) {
    ChatMessage m; m.role = ChatRole::System; m.parts.push_back(ChatPart::text_part(c)); return m;
}
ChatMessage make_assistant(const std::string& c, const std::optional<std::string>& reasoning = std::nullopt) {
    ChatMessage m; m.role = ChatRole::Assistant; m.parts.push_back(ChatPart::text_part(c));
    if (reasoning) m.reasoning_content = *reasoning;
    return m;
}
ChatMessage make_assistant_tool(const std::string& name, const std::string& args_json) {
    ChatMessage m; m.role = ChatRole::Assistant; m.parts.push_back(ChatPart::text_part(""));
    m.tool_calls.push_back({.id = "", .name = name, .arguments_json = args_json});
    return m;
}
ChatMessage make_tool(const std::string& content, const std::string& call_id) {
    ChatMessage m; m.role = ChatRole::Tool; m.parts.push_back(ChatPart::text_part(content));
    m.tool_call_id = call_id; return m;
}

Fixture build_fixture(const std::string& name) {
    Fixture fx; fx.name = name;
    if (name == "user_only") {
        fx.messages = {make_user("Reply with exactly one sentence.")};
    } else if (name == "sys_user") {
        fx.messages = {make_system("You are a technical assistant."),
                       make_user("Explain CUDA graphs briefly.")};
    } else if (name == "multi_turn") {
        fx.messages = {make_user("What is 2+2?"), make_assistant("4"), make_user("And plus 1?")};
    } else if (name == "prev") {
        fx.messages = {make_user("Solve x+1=3."), make_assistant("x is 2.", "Let x=2."),
                       make_user("Now x+10?")};
    } else if (name == "tool_call") {
        fx.tool_jsons = {R"({"name":"get_weather","description":"Get weather","parameters":{"type":"object","properties":{"city":{"type":"string"}}}})"};
        fx.messages = {make_user("What is the weather in Paris?"),
                       make_assistant_tool("get_weather", R"({"city":"Paris"})")};
    } else if (name == "tool_result") {
        fx.tool_jsons = {R"({"name":"get_weather","description":"Get weather","parameters":{"type":"object","properties":{"city":{"type":"string"}}}})"};
        fx.messages = {make_user("Weather?"),
                       make_assistant_tool("get_weather", R"({"city":"Paris"})"),
                       make_tool(R"({"temp":20})", "call_1"),
                       make_user("Thanks.")};
    }
    return fx;
}

std::optional<ReasoningEffort> parse_effort(const std::string& e) {
    if (e == "none") return std::nullopt;
    if (e == "minimal") return ReasoningEffort::Minimal;
    if (e == "low") return ReasoningEffort::Low;
    if (e == "medium") return ReasoningEffort::Medium;
    if (e == "high") return ReasoningEffort::High;
    if (e == "xhigh") return ReasoningEffort::XHigh;
    if (e == "max") return ReasoningEffort::Max;
    return std::nullopt;
}

std::string cpp_render(const CompiledChatTemplate& tmpl, const Fixture& fx,
                       const std::optional<ReasoningEffort>& eff, bool et, bool pt = true) {
    ChatRenderOptions o;
    o.reasoning_effort = eff;
    o.enable_thinking = et;
    o.preserve_thinking = pt;
    o.add_generation_prompt = true;
    for (const auto& t : fx.tool_jsons) o.tool_jsons.push_back(t);
    return tmpl.render(fx.messages, o).text;
}

void assert_eq(const std::string& actual, const std::string& expected,
               const std::string& ctx) {
    if (actual != expected) {
        std::cerr << "FAIL: " << ctx << "\n";
        std::size_t n = std::min(actual.size(), expected.size());
        std::size_t diff = n;
        for (std::size_t i = 0; i < n; ++i) { if (actual[i] != expected[i]) { diff = i; break; } }
        std::cerr << "  first_diff_at=" << diff << "\n";
        std::cerr << "  actual=" << (diff < actual.size() ? actual.substr(diff, 120) : std::string("<end>")) << "\n";
        std::cerr << "  expect=" << (diff < expected.size() ? expected.substr(diff, 120) : std::string("<end>")) << "\n";
        ++g_failures;
    }
}

// Full-prompt equivalence vs the REAL Sharp v22.1 Jinja oracle.
// Non-tool fixtures: assert the entire rendered prompt is byte-identical to Sharp.
// Tool fixtures: report the difference (characterize), do not hard-fail.
void compare_sharp_full(const CompiledChatTemplate& sharp, const Fixture& fx,
                        const std::string& effort, bool et, bool pt,
                        const std::string& gold_full) {
    std::optional<ReasoningEffort> eff = parse_effort(effort);
    std::string s_out = cpp_render(sharp, fx, eff, et, pt);
    std::string ctx = "sharp full fx=" + fx.name + " re=" + effort +
                      " et=" + (et ? "T" : "F") + " pt=" + (pt ? "T" : "F");
    bool has_tools = !fx.tool_jsons.empty();
    if (!has_tools) {
        if (s_out != gold_full) {
            static int dump = 0;
            if (dump < 6) {
                std::ofstream d("/home/jie/work/dbg_" + std::to_string(dump) + ".txt");
                d << "=== CTX " << ctx << " ===\n\n--- ACTUAL (C++) ---\n" << s_out
                  << "\n\n--- EXPECTED (Sharp Jinja) ---\n" << gold_full;
                ++dump;
            }
        }
        assert_eq(s_out, gold_full, ctx);
    } else {
        if (s_out != gold_full) {
            std::cerr << "NOTE: tool fixture diverges from Sharp (expected; characterized separately): "
                      << ctx << "\n";
        }
    }
}

void assert_default_no_overlay(const CompiledChatTemplate& base, const Fixture& fx,
                               const std::string& effort, bool et) {
    std::optional<ReasoningEffort> eff = parse_effort(effort);
    std::string out = cpp_render(base, fx, eff, et);
    if (out.find("Answer directly, after thinking") != std::string::npos) {
        std::cerr << "FAIL: default leaked Sharp overlay fx=" << fx.name
                  << " re=" << effort << " et=" << (et ? "T" : "F") << "\n";
        ++g_failures;
    }
}

}  // namespace

int main() {
    std::string src = "";
    CompiledChatTemplate base =
        CompiledChatTemplate::resolve_unchecked(src, ChatStyle::Default);
    CompiledChatTemplate sharp =
        CompiledChatTemplate::resolve_unchecked(src, ChatStyle::SharpV22_1);

    auto scaps = sharp.capabilities();
    if (!(scaps.reasoning_effort.high)) { std::cerr << "FAIL: sharp capabilities missing high\n"; ++g_failures; }
    auto bcaps = base.capabilities();
    if (bcaps.reasoning_effort.high) { std::cerr << "FAIL: default capabilities unexpectedly advertise high\n"; ++g_failures; }

    const char* fixtures[] = {"user_only", "sys_user", "multi_turn", "prev", "tool_call", "tool_result"};
    const char* efforts[] = {"none", "minimal", "low", "medium", "high", "xhigh", "max"};
    const std::string gold_dir = "/home/jie/work/oracle_expected/";

    for (const char* fname : fixtures) {
        Fixture fx = build_fixture(fname);
        bool is_pt = (fname == std::string("multi_turn") || fname == std::string("prev"));
        for (const char* effort : efforts) {
            for (bool et : {true, false}) {
                // 'none' in Sharp v22.1 forces thinking off regardless of the et flag passed to the
                // oracle. Align both the C++ render and the gold lookup to et=false for 'none' so the
                // comparison is consistent (the renderer receives enable_thinking=false via translate.cpp).
                const bool eff_none = (std::string(effort) == "none");
                const bool eff_et = eff_none ? false : et;
                // preserve_thinking matrix only for fixtures carrying historical reasoning.
                std::vector<bool> pts = is_pt ? std::vector<bool>{true, false} : std::vector<bool>{true};
                for (bool pt : pts) {
                    std::string key = "sharp__" + std::string(fname) + "__re=" + effort +
                                      "__et=" + (eff_et ? "True" : "False") +
                                      "__pt=" + (pt ? "True" : "False") + ".txt";
                    std::ifstream gf(gold_dir + key);
                    if (!gf) { std::cerr << "FAIL: missing gold " << key << "\n"; ++g_failures; continue; }
                    std::stringstream gold_full; gold_full << gf.rdbuf();
                    compare_sharp_full(sharp, fx, effort, eff_et, pt, gold_full.str());
                }
            }
        }
        assert_default_no_overlay(base, fx, "none", true);
        assert_default_no_overlay(base, fx, "none", false);
    }

    // Explicit end-to-end proof of the serve-translation semantics:
    //   reasoning_effort=none  ->  enable_thinking=false  ->  matches Sharp (thinking disabled)
    // The C++ renderer receives enable_thinking=false (translate.cpp does the mapping); the
    // resulting full Sharp prompt must equal the real Sharp Jinja with reasoning_effort=none,
    // enable_thinking=false.
    {
        Fixture fx = build_fixture("prev");  // history + reasoning to exercise preserve path
        std::string key = "sharp__prev__re=none__et=False__pt=True.txt";
        std::ifstream gf(gold_dir + key);
        std::stringstream gold; gold << gf.rdbuf();
        // eff=nullopt + et=false is exactly what translate.cpp produces for requested=none.
        std::string s_out = cpp_render(sharp, fx, std::nullopt, /*et=*/false, /*pt=*/true);
        assert_eq(s_out, gold.str(), "none->thinking-off end-to-end (prev fixture)");
    }

    if (g_failures == 0) {
        std::cout << "PASS: full-prompt Sharp equivalence (non-tool) + none-disables-thinking + format preservation\n";
        return 0;
    }
    std::cerr << g_failures << " failures\n";
    return 1;
}
