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
  bool is_root;
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
      body_ << std::to_string(f) << "f";
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

  void visit(RandStmt *stmt) override {
    init_rand();
    auto dt = stmt->element_type();
    auto dt_name = get_primitive_type_name(dt);
    emit_let(stmt->raw_name(),dt_name);
    if(dt->is_primitive( PrimitiveTypeID::i32)){
      body_ << "rand_i32(gid3.x);\n";
    }
    else if(dt->is_primitive( PrimitiveTypeID::f32)){
      body_ << "rand_f32(gid3.x);\n";
    }
    else{
      TI_ERROR("unsupported prim type in rand")
    }
  }

  void visit(UnaryOpStmt *stmt) override {
    const auto operand = stmt->operand->raw_name();

    const auto src_dt = stmt->operand->element_type();
    const auto dst_dt = stmt->element_type();
    std::string dst_dt_name = get_primitive_type_name(dst_dt);
    UnaryOpType op = stmt->op_type;
    
    std::string value;
    if(false){

    }
#define HANDLE_FUNC_OP(op_name, func)\
    else if(op == UnaryOpType::op_name){ \
      value = std::string(func) + "("+operand+")"; \
    }
    HANDLE_FUNC_OP(sqrt,"sqrt")
    HANDLE_FUNC_OP(round,"round")
    HANDLE_FUNC_OP(floor,"floor")
    HANDLE_FUNC_OP(ceil,"ceil")
    HANDLE_FUNC_OP(abs,"abs")
    HANDLE_FUNC_OP(sgn,"sign")
    HANDLE_FUNC_OP(sin,"sin")
    HANDLE_FUNC_OP(asin,"asin")
    HANDLE_FUNC_OP(cos,"cos")
    HANDLE_FUNC_OP(acos,"acos")
    HANDLE_FUNC_OP(tan,"tan")
    HANDLE_FUNC_OP(tanh,"tanh")
    HANDLE_FUNC_OP(tan,"inv")
    HANDLE_FUNC_OP(exp,"exp")
    HANDLE_FUNC_OP(log,"log")
    HANDLE_FUNC_OP(rsqrt,"inverseSqrt")
#undef HANDLE_FUNC_OP
    else if(op == UnaryOpType::neg){
      value = "-"+operand;
    }
    else if(op == UnaryOpType::logic_not){
      std::string zero ;
      if(src_dt->is_primitive(PrimitiveTypeID::f32)){
        zero = "0.0f";
      }
      else if(src_dt->is_primitive(PrimitiveTypeID::i32)){
        zero = "0";
      }
      else{
        TI_ERROR("unsupported prim type in unary op");
      } 
      value = std::string(dst_dt_name)+"(" + operand + " == "+zero+")"; 
    }
    else if(op == UnaryOpType::bit_not){
      value = "~" + operand;
    }
    else if(op == UnaryOpType::inv || op == UnaryOpType::rcp){
      value = "(1.0f / f32("+operand+"))";
    }
    else if(op == UnaryOpType::cast_value){
      value = std::string(dst_dt_name)+"(" + operand + ")"; 
    }
    else if(op == UnaryOpType::cast_bits){
      value = "bitcast<"+std::string(dst_dt_name)+">(" + operand + ")"; 
    }
    emit_let(stmt->raw_name(),dst_dt_name);
    value = std::string(dst_dt_name) + "(" + value + ")";
    body_ << value<<";\n";
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
#define HANDLE_INFIX_OP_U32(op_name, infix_token)\
    else if(op == BinaryOpType::op_name){ \
      value = lhs + " "+ infix_token + " u32(" + rhs+")"; \
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
    HANDLE_INFIX_OP_U32(bit_shl, "<<")
    HANDLE_INFIX_OP_U32(bit_shr, ">>") // TODO: fix
    HANDLE_INFIX_OP_U32(bit_sar, ">>") // TODO: fix
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
    
    value = std::string(get_primitive_type_name(dt)) + "(" + value + ")";

    body_ << value<<";\n";
  }

  
  void visit(RangeForStmt *stmt) override {
    TI_ASSERT(stmt->width() == 1);
    emit_var(stmt->raw_name(),"i32");
    body_<< stmt->begin->raw_name() <<";\n";
    body_ << body_indent() << "loop {\n";
    indent();
    body_ << body_indent() << "if ("<< stmt->raw_name() << " >= "<<stmt->end->raw_name() << "){ break; }\n";

    stmt->body->accept(this);

    body_ << body_indent() << stmt->raw_name() << " = "<< stmt->raw_name() << " + 1;\n";
    dedent();
    body_ << body_indent() << "}\n";
  }

  void visit(IfStmt *stmt) override {
    body_ << body_indent() << "if (bool("<< stmt->cond->raw_name() << ")){\n";
    indent();

    if (stmt->true_statements) {
      stmt->true_statements->accept(this);
    }

    dedent();
    body_ << body_indent() << "}\n";
    if (stmt->false_statements) {
      body_ << body_indent() << "else {\n";
      indent();
      stmt->false_statements->accept(this);
      dedent();
      body_ << body_indent() << "}\n";
    }
  }

  void visit(WhileControlStmt *stmt) override {
    body_ <<  body_indent() << "break;\n";
  }

  void visit(ContinueStmt *stmt) override {
    auto stmt_in_off_for = [stmt]() {
      TI_ASSERT(stmt->scope != nullptr);
      if (auto *offl = stmt->scope->cast<OffloadedStmt>(); offl) {
        TI_ASSERT(offl->task_type == OffloadedStmt::TaskType::range_for ||
                  offl->task_type == OffloadedStmt::TaskType::struct_for);
        return true;
      }
      return false;
    };
    if(stmt_in_off_for()){
      // then this parallel task is done; 
      body_ << body_indent() << "ii = ii + total_invocs;\n";
      // continue in grid-strided loop;
      body_ <<  body_indent() << "continue;\n";
    }
    else{
      body_ <<  body_indent() << "continue;\n";
    }
  }

  void visit(WhileStmt *stmt) override {
    body_ << body_indent() << "loop {\n";
    indent();
    stmt->body->accept(this);
    dedent();
    body_ << body_indent() << "}\n";
  }

  void visit(ArgLoadStmt *stmt) override {
    const auto arg_id = stmt->arg_id;
    const auto &arg_attribs = ctx_attribs_->args()[arg_id];
    const auto offset_in_mem = arg_attribs.offset_in_mem;
    if (stmt->is_ptr) {
      TI_ERROR("arg is ptr... what does that mean lol");
    } else {
      const auto dt = arg_attribs.dt;
      std::string buffer_name = get_buffer_member_name(BufferInfo(BufferType::Args));
      emit_let(stmt->raw_name(), get_primitive_type_name(dt));
      body_ << "bitcast<" << get_primitive_type_name(dt)<<">(" <<buffer_name << "[" << std::to_string(offset_in_mem/4) <<"]);\n";
    }
  }

  void visit(AllocaStmt *stmt) override {
    auto dt = stmt->element_type();
    // not using emit_var() because it emits an extra equals token..
    body_ << body_indent() << "var "<<stmt->raw_name() << " : "<<get_primitive_type_name(dt)<<";\n";
  }

  void visit(LocalLoadStmt *stmt) override {
    TI_ASSERT(stmt->src.size() == 1);
    TI_ASSERT(stmt->src[0].offset == 0);
    auto dt = stmt->element_type();
    emit_let(stmt->raw_name(),get_primitive_type_name(dt));
    body_ << stmt->src[0].var->raw_name() << ";\n";
  }

  void visit(LocalStoreStmt *stmt) override {
    body_ << body_indent()<< stmt->dest->raw_name() <<" = "<<stmt->val->raw_name()<<";\n";
  }

  void visit(GetRootStmt *stmt) override {
    const int root_id = snode_to_root_.at(stmt->root()->id);
    emit_let(stmt->raw_name(), get_pointer_int_type_name());
    body_ << "0;\n";
    pointer_infos_[stmt->raw_name()] = {true, root_id};
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
    pointer_infos_[stmt->raw_name()] = {true, root};
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
    pointer_infos_[stmt->raw_name()] = {true, root_id};
  }

  void visit(GlobalTemporaryStmt *stmt) override {
    TI_ASSERT(stmt->width() == 1);
    emit_let(stmt->raw_name(), get_pointer_int_type_name());
    body_ << (stmt->offset) / 4 << ";\n";
    pointer_infos_[stmt->raw_name()] = {false, -1};
  }

  void visit(GlobalStoreStmt *stmt) override {
    TI_ASSERT(stmt->width() == 1);
    PointerInfo info = pointer_infos_.at(stmt->dest->raw_name());
    std::string buffer_name;
    if(info.is_root){
      int root_id = info.root_id;
      buffer_name = get_buffer_member_name(BufferInfo(BufferType::RootNormal, root_id));
    }
    else{
      buffer_name = get_buffer_member_name(BufferInfo(BufferType::GlobalTemps));
    }
    body_ << body_indent() << buffer_name << "[" << stmt->dest->raw_name() <<"] = bitcast<i32>("<<(stmt->val->raw_name())<<");\n";
  }

  void visit(GlobalLoadStmt *stmt) override {
    TI_ASSERT(stmt->width() == 1);
    auto dt = stmt->element_type();
    PointerInfo info = pointer_infos_.at(stmt->src->raw_name());
    std::string buffer_name;
    if(info.is_root){
      int root_id = info.root_id;
      buffer_name = get_buffer_member_name(BufferInfo(BufferType::RootNormal, root_id));
    }
    else{
      buffer_name = get_buffer_member_name(BufferInfo(BufferType::GlobalTemps));
    }
    emit_let(stmt->raw_name(), get_primitive_type_name(dt));
    body_ << "bitcast<" << get_primitive_type_name(dt)<<">(" <<buffer_name << "[" << stmt->src->raw_name() <<"]);\n";
  }

  void visit(AtomicOpStmt *stmt) override {
    TI_ASSERT(stmt->width() == 1);
    const auto dt = stmt->dest->element_type().ptr_removed();
    std::string dt_name = get_primitive_type_name(dt);
    std::string buffer_member_name;
    PointerInfo info = pointer_infos_.at(stmt->dest->raw_name());
    TI_ASSERT(info.is_root);
    int root_id = info.root_id;
    buffer_member_name = get_buffer_member_name(BufferInfo(BufferType::RootAtomicI32, root_id));
    std::string atomic_func_name;
    switch(stmt->op_type){
      case AtomicOpType::add:{
        atomic_func_name = "atomicAdd";
        break;
      }
      case AtomicOpType::sub:{
        atomic_func_name = "atomicSub";
        break;
      }
      case AtomicOpType::max:{
        atomic_func_name = "atomicMax";
        break;
      }
      case AtomicOpType::min:{
        atomic_func_name = "atomicMin";
        break;
      }
      case AtomicOpType::bit_and:{
        atomic_func_name = "atomicAnd";
        break;
      }
      case AtomicOpType::bit_or:{
        atomic_func_name = "atomicOr";
        break;
      }
      case AtomicOpType::bit_xor:{
        atomic_func_name = "atomicXor";
        break;
      }
    }

    /*
fn atomicAddFloat(dest: ptr<storage, atomic<i32>, read_write>, v: f32) -> f32 {
  loop {
    let old_val : f32 = bitcast<f32>(atomicLoad(dest));
    let new_val : f32 = old_val + v;
    if(atomicCompareExchangeWeak(dest, bitcast<i32>(old_val), bitcast<i32>(new_val)).y != 0){
      return old_val;
    }
  }
}
*/
    
    // WGSL doesn't allow declaring a function whose argument is a pointer to SSBO... so we inline it

    auto result = get_temp("atomic_op_result");

    body_<<body_indent()<<"var " << result << " : "<< dt_name<<";\n";

    std::string ptr = "&(" + buffer_member_name + "[" + stmt->dest->raw_name() + "])";

    if(dt->is_primitive(PrimitiveTypeID::i32)){
      body_<<body_indent()<< result << " = "<< atomic_func_name << "(" << ptr << ", "<<stmt->val->raw_name()<<");\n";
    }
    else if(dt->is_primitive(PrimitiveTypeID::f32)){
      body_<<body_indent()<<"loop {\n";
      indent();
      std::string old_val = get_temp("old_val");
      emit_let(old_val,"f32");
      body_<< "bitcast<f32>(atomicLoad("<<ptr<<"));\n";

      std::string new_val_expr;
       switch(stmt->op_type){
        case AtomicOpType::add:{
          new_val_expr = old_val +" + "+stmt->val->raw_name();
          break;
        }
        case AtomicOpType::sub:{
          new_val_expr = old_val +" - "+stmt->val->raw_name();
          break;
        }
        case AtomicOpType::max:{
          new_val_expr = "max(" +old_val +", "+stmt->val->raw_name()+ ")";
          break;
        }
        case AtomicOpType::min:{
          new_val_expr = "min(" +old_val +", "+stmt->val->raw_name()+ ")";
          break;
        }
        default:
        TI_ERROR("unsupported atomic op for f32");
      }

      std::string new_val = get_temp("new_val");
      emit_let(new_val,"f32");
      body_<< new_val_expr <<";\n";
      body_<<body_indent()<<"if(atomicCompareExchangeWeak("<<ptr<< ", bitcast<i32>("<<  old_val << "), bitcast<i32>("<<new_val<<")).y!=0){\n";
      indent();
      body_<<body_indent() << result << " = "<<old_val<<";\n";
      body_<<body_indent()<<"break;\n";
      dedent();
      body_<<body_indent()<<"}\n";
      dedent();
      body_<<body_indent()<<"}\n";
    }
    else{
      TI_ERROR("unsupported prim type in atomic op")
    }
    //body_<<body_indent()<<"storageBarrier();\n";
    emit_let(stmt->raw_name(),dt_name);
    body_<< result <<";\n";
  }


  void visit(LoopIndexStmt *stmt) override {
    const auto stmt_name = stmt->raw_name();
    if (stmt->loop->is<OffloadedStmt>()) {
      const auto type = stmt->loop->as<OffloadedStmt>()->task_type;
      if (type == OffloadedTaskType::range_for) {
        TI_ASSERT(stmt->index == 0);
        emit_let(stmt->raw_name(), "i32");
        body_ << "ii;\n";
      } else {
        TI_NOT_IMPLEMENTED;
      }
    } else if (stmt->loop->is<RangeForStmt>()) {
      TI_ASSERT(stmt->index == 0);
      emit_let(stmt->raw_name(), "i32");
      body_ << stmt->loop->raw_name() <<";\n";
    } else {
      TI_NOT_IMPLEMENTED;
    }
  }

 private: 

  void emit_let(std::string name, std::string type){
    body_ << body_indent() << "let " << name << " : "<<type<<" = ";
  }

  void emit_var(std::string name, std::string type){
    body_ << body_indent() << "var " << name << " : "<<type<<" = ";
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
     task_attribs_.name = task_name_;
    task_attribs_.task_type = OffloadedTaskType::range_for;
    // task_attribs_.buffer_binds = get_common_buffer_binds();

    task_attribs_.range_for_attribs = TaskAttributes::RangeForAttributes();
    auto &range_for_attribs = task_attribs_.range_for_attribs.value();
    range_for_attribs.const_begin = stmt->const_begin;
    range_for_attribs.const_end = stmt->const_end;
    range_for_attribs.begin =
        (stmt->const_begin ? stmt->begin_value : stmt->begin_offset);
    range_for_attribs.end =
        (stmt->const_end ? stmt->end_value : stmt->end_offset);
    int block_size = stmt->block_dim;
    task_attribs_.advisory_num_threads_per_group = block_size;
    start_function(block_size);

    std::string total_elems_value;
    std::string begin_expr_value;  
    std::string end_expr_value;
    if (range_for_attribs.const_range()) {
      const int num_elems = range_for_attribs.end - range_for_attribs.begin;
      begin_expr_value = std::to_string(stmt->begin_value);
      total_elems_value = std::to_string( num_elems);
      end_expr_value = std::to_string(stmt->begin_value + num_elems);
      task_attribs_.advisory_total_num_threads = num_elems;
    } else {
      if (!stmt->const_begin) {
        emit_let("begin_idx","i32");
        body_ << std::to_string(stmt->begin_offset / 4) << ";\n";

        std::string gtmps_buffer_member_name = get_buffer_member_name(BufferInfo(BufferType::GlobalTemps));
        begin_expr_value = gtmps_buffer_member_name + "[begin_idx]";
      } else {
        begin_expr_value = std::to_string(stmt->begin_value); 
      }
      
      if (!stmt->const_end) {
        emit_let("end_idx","i32");
        body_ << std::to_string(stmt->end_offset / 4) << ";\n";
        std::string gtmps_buffer_member_name = get_buffer_member_name(BufferInfo(BufferType::GlobalTemps));
        end_expr_value = gtmps_buffer_member_name + "[end_idx]";
      } else {
        end_expr_value = std::to_string(stmt->end_value); 
      }
      total_elems_value = "end_ - begin_";
      task_attribs_.advisory_total_num_threads = 65536;
    }
    emit_let("begin_","i32");
    body_ << begin_expr_value << ";\n";
    emit_let("end_","i32");
    body_<< body_indent()<< end_expr_value<<";\n";
    emit_let("total_elems","i32");
    body_ << total_elems_value << ";\n";

    emit_let("total_invocs","i32");
    body_ << block_size << " * i32(n_workgroups.x);\n";

    emit_var("ii", "i32");
    body_ << "i32(gid3.x) + begin_;\n";

    body_ << body_indent() << "loop {\n";
    indent();
    body_ << body_indent() << "if(ii >= end_) { break;  }\n";

    stmt->body->accept(this);

    body_ << body_indent() << "ii = ii + total_invocs;\n";
    dedent();
    body_ << body_indent() << "}\n";
  } 

  StringBuilder global_decls_;
  StringBuilder function_signature_;
  StringBuilder function_body_prologue_;
  StringBuilder body_;
  StringBuilder function_body_epilogue_;
  StringBuilder function_end_;

  std::string assemble_shader(){
    return 
    global_decls_.getString() +
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
fn main(
  [[builtin(global_invocation_id)]] gid3 : vec3<u32>, 
  [[builtin(num_workgroups)]] n_workgroups : vec3<u32>) 
{

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

  int next_internal_temp = 0;
  std::string get_temp(std::string hint = ""){
    return std::string("_internal_temp")+std::to_string(next_internal_temp++)+"_"+hint;
  }

  std::unordered_map<std::string, PointerInfo> pointer_infos_;

  std::string get_buffer_member_name(BufferInfo buffer){
    return get_buffer_name(buffer)+".member";
  }

  std::string get_buffer_name(BufferInfo buffer){
    std::string name;
    std::string element_type = "i32";
    std::string stride = "4";
    switch(buffer.type){
      case BufferType::RootNormal: {
        name = "root_buffer_"+std::to_string(buffer.root_id)+"_";
        break;
      }
      case BufferType::RootAtomicI32: {
        name = "root_buffer_"+std::to_string(buffer.root_id)+"_atomic_";
        element_type = "atomic<i32>";
        break;
      }
      case BufferType::GlobalTemps: {
        name = "global_tmps_";
        break;
      }
      case BufferType::RandStates: {
        name = "rand_states_";
        element_type = "RandState";
        stride = "16";
        break;
      }
      case BufferType::Args: {
        name = "args_";
        break;
      }
    }
    if(task_attribs_.buffer_bindings.find(buffer) == task_attribs_.buffer_bindings.end()){
      int binding = task_attribs_.buffer_bindings.size();
      task_attribs_.buffer_bindings[buffer] = binding;
      declare_new_buffer(buffer,name,binding,stride, element_type);
    }
    return name;
  }

  void declare_new_buffer(BufferInfo buffer, std::string name, int binding,std::string stride, std::string element_type){
    std::string decl_template =
R"(

[[block]]
struct BUFFER_TYPE_NAME {
    member: [[stride(STRIDE)]] array<ELEMENT_TYPE>;
};
[[group(0), binding(BUFFER_BINDING)]]
var<storage, read_write> BUFFER_NAME: BUFFER_TYPE_NAME;

)"; 

    string_replace_all(decl_template,"BUFFER_TYPE_NAME", name+"_type");
    string_replace_all(decl_template,"BUFFER_NAME", name);
    string_replace_all(decl_template,"BUFFER_BINDING", std::to_string(binding));
    string_replace_all(decl_template,"ELEMENT_TYPE", element_type);
    string_replace_all(decl_template,"STRIDE", stride);
    global_decls_ << decl_template;
  }


  bool rand_initiated_ = false;
  void init_rand(){
    if(rand_initiated_){
      return;
    }
    std::string struct_decl =
R"(

struct RandState{
  x: u32;
  y: u32;
  z: u32;
  w: u32;
};

)"; 
    global_decls_ << struct_decl;
    std::string rand_states_member_name = get_buffer_member_name(BufferInfo(BufferType::RandStates));
    std::string rand_func_decl = 
R"(

fn rand_u32(id: u32) -> u32 {
  var state : RandState = STATES[id];
  if(state.x == 0u && state.y == 0u && state.z == 0u && state.w == 0u){
    state.x = 123456789u * id * 1000000007u;
    state.y = 362436069u;
    state.z = 521288629u;
    state.w = 88675123u;
  }
  let t : u32 = state.x ^ (state.x << 11u);
  state.x = state.y;
  state.y = state.z;
  state.z = state.w;
  state.w = (state.w ^ (state.w >> 19u)) ^ (t ^ (t >> 8u)); 
  let result : u32 = state.w * 1000000007u;
  STATES[id] = state;
  return result;
}

fn rand_f32(id:u32) -> f32 {
  let u32_res : u32 = rand_u32(id);
  return f32(u32_res) * (1.0f / 4294967296.0f);
}

fn rand_i32(id:u32) -> i32 {
  let u32_res : u32 = rand_u32(id);
  return i32(u32_res);
}

)"; 
    string_replace_all(rand_func_decl,"STATES",rand_states_member_name);
    global_decls_ << rand_func_decl;
    rand_initiated_ = true;
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
