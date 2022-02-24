#include "taichi/backends/webgpu/wgsl_codegen.h"

#include <string>
#include <sstream>
#include <vector>

#include "taichi/program/program.h"
#include "taichi/program/kernel.h"
#include "taichi/ir/statements.h"
#include "taichi/ir/ir.h"
#include "taichi/util/line_appender.h"
#include "taichi/backends/webgpu/kernel_utils.h"
#include "taichi/ir/transforms.h"
#include "taichi/math/arithmetic.h"
 

namespace taichi {
namespace lang {
namespace webgpu {

using BufferInfo = TaskAttributes::BufferInfo;
using BufferType = TaskAttributes::BufferType;
namespace{

class StringBuilder {
public:
  const std::string& getString(){
    return buffer_;
  }

  template<typename T>
  StringBuilder& operator<< (const T& t){
    std::ostringstream os;
    os << t;
    buffer_ += os.str();
    return *this;
  }

private:
  std::string buffer_;
};

struct PointerInfo {
  int root_id;
};

void string_replace_all(std::string& str, const std::string& from, const std::string& to) {
    if(from.empty())
        return;
    size_t start_pos = 0;
    while((start_pos = str.find(from, start_pos)) != std::string::npos) {
        str.replace(start_pos, from.length(), to);
        start_pos += to.length(); // In case 'to' contains 'from', like replacing 'x' with 'yx'
    }
}
    
class TaskCodegen : public IRVisitor {
 public:
  struct Params {
    OffloadedStmt *task_ir;
    std::vector<CompiledSNodeStructs> compiled_structs;
    const KernelContextAttributes *ctx_attribs;
    std::string ti_kernel_name;
    int task_id_in_kernel;
  };

  const bool use_64bit_pointers = false;

  explicit TaskCodegen(const Params &params)
      : 
        task_ir_(params.task_ir),
        compiled_structs_(params.compiled_structs),
        ctx_attribs_(params.ctx_attribs),
        task_name_(fmt::format("{}_t{:02d}",
                               params.ti_kernel_name,
                               params.task_id_in_kernel)) {
    allow_undefined_visitor = true;
    invoke_default_visitor = true;

    fill_snode_to_root();
  }

  void fill_snode_to_root() {
    for (int root = 0; root < compiled_structs_.size(); ++root) {
      for (auto [node_id, node] : compiled_structs_[root].snode_descriptors) {
        snode_to_root_[node_id] = root;
      }
    }
  }

  struct Result {
    std::string wgsl_code;
    TaskAttributes task_attribs;
  };

  Result run() { 

    //compile_args_struct();

    if (task_ir_->task_type == OffloadedTaskType::serial) {
      generate_serial_kernel(task_ir_);
    } else if (task_ir_->task_type == OffloadedTaskType::range_for) {
      // struct_for is automatically lowered to ranged_for for dense snodes
      generate_range_for_kernel(task_ir_);
    } else if (task_ir_->task_type == OffloadedTaskType::struct_for) {
      generate_struct_for_kernel(task_ir_);
    } else {
      TI_ERROR("Unsupported offload type={} on WGSL codegen",
               task_ir_->task_name());
    } 
    Result res;
    res.wgsl_code = assemble_shader();
    res.task_attribs = std::move(task_attribs_);

    return res;
  }

  void visit(OffloadedStmt *) override {
    TI_ERROR("This codegen is supposed to deal with one offloaded task");
  }

  void visit(Block *stmt) override {
    for (auto &s : stmt->statements) {
      s->accept(this);
    }
  }  

  void visit(ConstStmt * stmt) override {
    TI_ASSERT(stmt->width() == 1);
    auto& const_val = stmt->val[0];
    auto dt = const_val.dt.ptr_removed();
    emit_let(stmt->raw_name(), get_primitive_type_name(dt));
    if (dt->is_primitive(PrimitiveTypeID::f32)){
      float f = const_val.val_float32();
      body_ << f;
    }
    else if (dt->is_primitive(PrimitiveTypeID::i32)){
      int i = const_val.val_int32();
      body_ << i;
    }
    else{
      TI_ERROR("unsupported const type");
    }
    
    body_ << ";\n";
  }

  void visit(BinaryOpStmt * stmt) override {
    const auto lhs = stmt->lhs->raw_name();
    const auto rhs = stmt->rhs->raw_name();
    const auto op = stmt->op_type;
    DataType dt = stmt->element_type();
    emit_let(stmt->raw_name(), get_primitive_type_name(dt));

    std::string value;
    if(false){

    }
#define HANDLE_INFIX_OP(op_name, infix_token)\
    else if(op == BinaryOpType::op_name){ \
      value = lhs + " "+ infix_token + " " + rhs; \
    }
#define HANDLE_FUNC_OP(op_name, func)\
    else if(op == BinaryOpType::op_name){ \
      value = std::string(func) + "("+lhs +", "+rhs+")"; \
    }
    HANDLE_INFIX_OP(mul, "*")
    HANDLE_INFIX_OP(add, "+")
    HANDLE_INFIX_OP(sub, "-")
    HANDLE_INFIX_OP(mod, "%")
    HANDLE_INFIX_OP(bit_and, "&")
    HANDLE_INFIX_OP(bit_or, "|")
    HANDLE_INFIX_OP(bit_xor, "^")
    HANDLE_INFIX_OP(bit_shl, "<<")
    HANDLE_INFIX_OP(bit_shr, ">>") // TODO: fix
    HANDLE_INFIX_OP(bit_sar, ">>") // TODO: fix
    HANDLE_INFIX_OP(cmp_lt, "<")
    HANDLE_INFIX_OP(cmp_le, "<=")
    HANDLE_INFIX_OP(cmp_gt, ">")
    HANDLE_INFIX_OP(cmp_ge, ">=")
    HANDLE_INFIX_OP(cmp_eq, "==")
    HANDLE_INFIX_OP(cmp_ne, "!=")
    HANDLE_FUNC_OP(pow,"pow")
    HANDLE_FUNC_OP(atan2,"atan2")
    HANDLE_FUNC_OP(max,"max")
    HANDLE_FUNC_OP(min,"min")
#undef HANDLE_INFIX_OP
#undef HANDLE_FUNC_OP
    else if(op == BinaryOpType :: div){
      value = lhs + " / " + rhs;
    }
    else if(op == BinaryOpType :: truediv){
      value = "1.0 * "+ lhs + " / " + rhs;
    }
    else if(op == BinaryOpType :: floordiv){
      value = "floor(1.0 * "+ lhs + " / " + rhs+")";
    }

    body_ <<value<<";\n";
  }

  void visit(GetRootStmt *stmt) override {
    const int root_id = snode_to_root_.at(stmt->root()->id);
    emit_let(stmt->raw_name(), get_pointer_int_type_name());
    body_ << "0;\n";
    pointer_infos[stmt->raw_name()] = {root_id};
  }
  void visit(GetChStmt *stmt) override {
    // TODO: GetChStmt -> GetComponentStmt ?
    const int root = snode_to_root_.at(stmt->input_snode->id);

    const auto &snode_descs = compiled_structs_[root].snode_descriptors;
    auto *out_snode = stmt->output_snode;
    TI_ASSERT(snode_descs.at(stmt->input_snode->id).get_child(stmt->chid) ==
              out_snode);

    const auto &desc = snode_descs.at(out_snode->id);
    emit_let(stmt->raw_name(), get_pointer_int_type_name());
    body_ << stmt->input_ptr->raw_name() <<" + "<< (desc.mem_offset_in_parent_cell/4)<<";\n";
    pointer_infos[stmt->raw_name()] = {root};
  }

  void visit(SNodeLookupStmt *stmt) override {
    // TODO: SNodeLookupStmt -> GetSNodeCellStmt ?
    bool is_root{false};  // Eliminate first root snode access
    const int root_id = snode_to_root_.at(stmt->snode->id);
    std::string parent;

    if (stmt->input_snode) {
      parent = stmt->input_snode->raw_name();
    } else {
      TI_ASSERT(root_stmts_.at(root_id) != nullptr);
      parent = root_stmts_.at(root_id)->raw_name();
    }
    const auto *sn = stmt->snode;
 
    const auto &snode_descs = compiled_structs_[root_id].snode_descriptors;
    const auto &desc = snode_descs.at(sn->id);
    emit_let(stmt->raw_name(), get_pointer_int_type_name());
    body_ << parent <<" + ("<< (desc.cell_stride/4) <<" * "<< (stmt->input_index->raw_name()) << ");\n";
    pointer_infos[stmt->raw_name()] = {root_id};
  }

  void visit(GlobalStoreStmt *stmt) override {
    TI_ASSERT(stmt->width() == 1);
    int root_id = pointer_infos.at(stmt->dest->raw_name()).root_id;
    std::string buffer_name = get_buffer_member_name(BufferInfo(BufferType::RootNormal, root_id));
    body_ << body_indent() << buffer_name << "[" << stmt->dest->raw_name() <<"] = bitcast<i32>("<<(stmt->val->raw_name())<<");\n";
  }

 private: 

  void emit_let(std::string name, std::string type){
    body_ << body_indent() << "let " << name << " : "<<type<<" = ";
  }

  const char* get_pointer_int_type_name(){
    return "i32";
  }

  const char* get_primitive_type_name(DataType dt){
    if (dt->is_primitive(PrimitiveTypeID::f32)){
      return "f32";
    }
    if (dt->is_primitive(PrimitiveTypeID::i32)){
      return "i32";
    }
    TI_ERROR("unsupported primitive type");
    return "";
  }

  void generate_serial_kernel(OffloadedStmt *stmt) {
    task_attribs_.name = task_name_;
    task_attribs_.task_type = OffloadedTaskType::serial;
    // task_attribs_.buffer_binds = get_common_buffer_binds();
    task_attribs_.advisory_total_num_threads = 1;
    task_attribs_.advisory_num_threads_per_group = 1;

    start_function(1);
    stmt->body->accept(this);
  }

  void generate_range_for_kernel(OffloadedStmt *stmt) {
     
  }
 
  void generate_struct_for_kernel(OffloadedStmt *stmt) {
    
  }

  StringBuilder buffer_decls_;
  StringBuilder function_signature_;
  StringBuilder function_body_prologue_;
  StringBuilder body_;
  StringBuilder function_body_epilogue_;
  StringBuilder function_end_;

  std::string assemble_shader(){
    return 
    buffer_decls_.getString() +
    function_signature_.getString() +
    function_body_prologue_.getString() +
    body_ .getString()+
    function_body_epilogue_ .getString() + 
    function_end_.getString();
  }

  void start_function(int block_size_x){
    TI_ASSERT(function_signature_.getString().size()==0);
    std::string signature_template = 
R"(

[[stage(compute), workgroup_size(BLOCK_SIZE_X, 1, 1)]]
fn main([[builtin(global_invocation_id)]] gid3 : vec3<u32>) {

)";
    string_replace_all(signature_template,"BLOCK_SIZE_X",std::to_string(block_size_x));
    function_signature_ << signature_template;
    function_end_ << "\n}\n";

  }

  int body_indent_count_ = 1;
  void indent(){
    body_indent_count_++;
  }
  void dedent(){
    body_indent_count_--;
  }
  std::string body_indent(){
    return std::string(body_indent_count_ * 2, ' ');
  }

  std::unordered_map<std::string, PointerInfo> pointer_infos;

  std::string get_buffer_member_name(BufferInfo buffer){
    return get_buffer_name(buffer)+".member";
  }

  std::string get_buffer_name(BufferInfo buffer){
    std::string name;
    switch(buffer.type){
      case BufferType::RootNormal: {
        name = "root_buffer_"+std::to_string(buffer.root_id)+"_";
        break;
      }
    }
    if(task_attribs_.buffer_bindings.find(buffer) == task_attribs_.buffer_bindings.end()){
      int binding = task_attribs_.buffer_bindings.size();
      task_attribs_.buffer_bindings[buffer] = binding;
      declare_new_buffer(buffer,name,binding);
    }
    return name;
  }

  void declare_new_buffer(BufferInfo buffer, std::string name, int binding){
    switch(buffer.type){
      case BufferType::RootNormal: {
        std::string decl_template =
R"(

[[block]]
struct BUFFER_TYPE_NAME {
    member: [[stride(4)]] array<i32>;
};
[[group(0), binding(BUFFER_BINDING)]]
var<storage, read_write> BUFFER_NAME: BUFFER_TYPE_NAME;

)";
        string_replace_all(decl_template,"BUFFER_TYPE_NAME", name+"_type");
        string_replace_all(decl_template,"BUFFER_NAME", name);
        string_replace_all(decl_template,"BUFFER_BINDING", std::to_string(binding));
        buffer_decls_ << decl_template;
        break;
      }
    }
  }
   

  OffloadedStmt *const task_ir_;  // not owned
  std::vector<CompiledSNodeStructs> compiled_structs_;
  std::unordered_map<int, int> snode_to_root_;
  const KernelContextAttributes *const ctx_attribs_;  // not owned
  const std::string task_name_; 

  TaskAttributes task_attribs_;
  std::unordered_map<int, GetRootStmt *>
      root_stmts_;  // maps root id to get root stmt 
};

}
   
KernelCodegen::KernelCodegen(const Params &params)
    : params_(params), ctx_attribs_(*params.kernel) {
 
}

void KernelCodegen::run(TaichiKernelAttributes &kernel_attribs,
                        std::vector<std::string> &generated_wgsl) {
  auto *root = params_.kernel->ir->as<Block>();
  auto &tasks = root->statements;
  for (int i = 0; i < tasks.size(); ++i) {
    TaskCodegen::Params tp;
    tp.task_ir = tasks[i]->as<OffloadedStmt>();
    tp.task_id_in_kernel = i;
    tp.compiled_structs = params_.compiled_structs;
    tp.ctx_attribs = &ctx_attribs_;
    tp.ti_kernel_name = params_.ti_kernel_name;

    TaskCodegen cgen(tp);

    auto task_res = cgen.run();

    kernel_attribs.tasks_attribs.push_back(std::move(task_res.task_attribs));
    generated_wgsl.push_back(std::move(task_res.wgsl_code));
  }
  kernel_attribs.ctx_attribs = std::move(ctx_attribs_);
  kernel_attribs.name = params_.ti_kernel_name;
  kernel_attribs.is_jit_evaluator = params_.kernel->is_evaluator;
}

void lower(Kernel *kernel) {
  auto &config = kernel->program->config;
  config.demote_dense_struct_fors = true;
  irpass::compile_to_executable(kernel->ir.get(), config, kernel, kernel->grad,
                                /*ad_use_stack=*/false, config.print_ir,
                                /*lower_global_access=*/true,
                                /*make_thread_local=*/false);
}

}  // namespace spirv
}  // namespace lang
}  // namespace taichi
